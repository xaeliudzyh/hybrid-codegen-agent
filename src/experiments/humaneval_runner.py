"""
HumanEval benchmark runner.

Mode 1 (engine): prompt -> generate -> exec tests -> pass@k.
Mode 2 (agent): prompt -> agent pipeline (with optional speculative execution) -> exec tests -> pass@k + latency metrics.
"""

import json
import time
import math
import signal
import traceback
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

from engines.base import GenerativeEngine, GenerationResult
from agent.code_gen_agent import CodeGenAgent, AgentResult


HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"

@dataclass
class HumanEvalTask:
    task_id: str
    prompt: str
    canonical_solution: str
    test: str
    entry_point: str


def load_humaneval(path: Optional[str] = None) -> list[HumanEvalTask]:
    """Load tasks from .jsonl/.jsonl.gz. Downloads automatically if missing."""
    if path is None:
        here = Path(__file__).parent
        for candidate in ("humaneval_data.jsonl", "humaneval_data.jsonl.gz"):
            p = here / candidate
            if p.exists():
                path = str(p)
                break

    if path is None:
        path = str(_download_humaneval())

    tasks: list[HumanEvalTask] = []
    open_fn = open
    if path.endswith(".gz"):
        import gzip
        open_fn = gzip.open

    with open_fn(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tasks.append(HumanEvalTask(
                task_id=obj["task_id"],
                prompt=obj["prompt"],
                canonical_solution=obj["canonical_solution"],
                test=obj["test"],
                entry_point=obj["entry_point"],
            ))
    return tasks


def _download_humaneval() -> Path:
    import urllib.request
    dest = Path(__file__).parent / "humaneval_data.jsonl.gz"
    print(f"Downloading HumanEval dataset to {dest} ...")
    urllib.request.urlretrieve(HUMANEVAL_URL, str(dest))
    print("Done.")
    return dest


@dataclass
class TaskResult:
    task_id: str
    passed: bool
    completion: str
    error: Optional[str] = None
    generation_time_s: float = 0.0
    detection_step: Optional[int] = None
    total_steps: Optional[int] = None
    speculative_hit: Optional[bool] = None
    time_saved: Optional[float] = None


@dataclass
class BenchmarkResult:
    engine_type: str
    model_name: str
    remasking: Optional[str]
    mode: str                              # "engine" or "agent"
    k: int
    n_tasks: int
    results: list[list[TaskResult]] = field(default_factory=list)
    pass_at_k: Optional[float] = None
    mean_generation_time_s: Optional[float] = None


_EXEC_TIMEOUT = 10


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


def check_correctness(completion: str, task: HumanEvalTask, timeout: int = _EXEC_TIMEOUT) -> tuple[bool, Optional[str]]:
    code = task.prompt + completion + "\n" + task.test + "\n" + f"check({task.entry_point})"

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout)
    try:
        exec_globals: dict = {}
        exec(code, exec_globals)
        return True, None
    except _Timeout:
        return False, "timeout"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021): 1 - C(n-c,k)/C(n,k)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


COMPLETION_SYSTEM_PROMPT = "Complete the following Python function. Output ONLY the function body, no explanation."


def run_engine_mode(
    engine: GenerativeEngine,
    tasks: list[HumanEvalTask],
    k: int = 1,
    max_tokens: int = 512,
    temperature: float = 0.2,
    system_prompt: Optional[str] = None,
) -> BenchmarkResult:
    """Generate k completions per task, run tests, compute pass@k."""
    sys_prompt = system_prompt or COMPLETION_SYSTEM_PROMPT
    remasking = getattr(engine, "_remasking", None)

    bench = BenchmarkResult(
        engine_type=engine.engine_type,
        model_name=engine.model_name,
        remasking=remasking,
        mode="engine",
        k=k,
        n_tasks=len(tasks),
    )

    for ti, task in enumerate(tasks):
        prompt = f"{sys_prompt}\n\n{task.prompt}"
        samples: list[TaskResult] = []
        for si in range(k):
            t0 = time.perf_counter()
            result: GenerationResult = engine.generate(
                prompt, max_tokens=max_tokens, temperature=temperature,
            )
            gen_time = time.perf_counter() - t0
            completion = result.text
            passed, err = check_correctness(completion, task)
            samples.append(TaskResult(
                task_id=task.task_id,
                passed=passed,
                completion=completion,
                error=err,
                generation_time_s=gen_time,
            ))
        bench.results.append(samples)
        n_passed = sum(1 for s in samples if s.passed)
        status = f"PASS" if n_passed > 0 else "FAIL"
        print(f"  [{ti+1:3d}/{len(tasks)}] {task.task_id:<20s}  {status}  ({n_passed}/{k})")

    _compute_aggregate(bench)
    return bench


AGENT_TASK_TEMPLATE = "Complete the following Python function and execute it to verify:\n\n{prompt}"


def run_agent_mode(
    agent: CodeGenAgent,
    tasks: list[HumanEvalTask],
    k: int = 1,
    max_tokens: int = 512,
    max_iterations: int = 3,
    temperature: Optional[float] = None,
    speculative: bool = False,
    min_step_ratio: float = 0.1,
    check_interval: int = 1,
) -> BenchmarkResult:
    """Run tasks through the agent pipeline, collect pass@k and speculative metrics."""
    engine = agent.engine
    remasking = getattr(engine, "_remasking", None)

    bench = BenchmarkResult(
        engine_type=engine.engine_type,
        model_name=engine.model_name,
        remasking=remasking,
        mode="agent" + ("+speculative" if speculative else ""),
        k=k,
        n_tasks=len(tasks),
    )

    for ti, task in enumerate(tasks):
        task_text = AGENT_TASK_TEMPLATE.format(prompt=task.prompt)
        samples: list[TaskResult] = []
        for si in range(k):
            t0 = time.perf_counter()
            try:
                if speculative:
                    agent_result: AgentResult = agent.run_with_speculative_execution(
                        task_text,
                        max_iterations=max_iterations,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        min_step_ratio=min_step_ratio,
                        check_interval=check_interval,
                    )
                else:
                    agent_result = agent.run(
                        task_text,
                        max_iterations=max_iterations,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
            except Exception as exc:
                samples.append(TaskResult(
                    task_id=task.task_id,
                    passed=False,
                    completion="",
                    error=f"agent error: {exc}",
                    generation_time_s=time.perf_counter() - t0,
                ))
                continue

            gen_time = time.perf_counter() - t0
            completion = agent_result.generated_code
            passed, err = check_correctness(completion, task)

            det_step = tot_steps = spec_hit = t_saved = None
            if agent_result.metrics.iterations:
                it0 = agent_result.metrics.iterations[0]
                det_step = it0.speculative_detection_step
                tot_steps = it0.speculative_total_steps
                spec_hit = it0.speculative_hit
                t_saved = it0.speculative_time_saved

            samples.append(TaskResult(
                task_id=task.task_id,
                passed=passed,
                completion=completion,
                error=err,
                generation_time_s=gen_time,
                detection_step=det_step,
                total_steps=tot_steps,
                speculative_hit=spec_hit,
                time_saved=t_saved,
            ))
        bench.results.append(samples)
        n_passed = sum(1 for s in samples if s.passed)
        status = "PASS" if n_passed > 0 else "FAIL"
        print(f"  [{ti+1:3d}/{len(tasks)}] {task.task_id:<20s}  {status}  ({n_passed}/{k})")

    _compute_aggregate(bench)
    return bench


def _compute_aggregate(bench: BenchmarkResult) -> None:
    k = bench.k
    total_pass_at_k = 0.0
    total_time = 0.0
    n_samples = 0

    for samples in bench.results:
        n = len(samples)
        c = sum(1 for s in samples if s.passed)
        total_pass_at_k += pass_at_k(n, c, min(k, n))
        for s in samples:
            total_time += s.generation_time_s
            n_samples += 1

    bench.pass_at_k = total_pass_at_k / len(bench.results) if bench.results else 0.0
    bench.mean_generation_time_s = total_time / n_samples if n_samples else 0.0


def save_results(bench: BenchmarkResult, path: str) -> None:
    data = {
        "engine_type": bench.engine_type,
        "model_name": bench.model_name,
        "remasking": bench.remasking,
        "mode": bench.mode,
        "k": bench.k,
        "n_tasks": bench.n_tasks,
        "pass_at_k": bench.pass_at_k,
        "mean_generation_time_s": bench.mean_generation_time_s,
        "tasks": [],
    }
    for samples in bench.results:
        task_entry = {
            "task_id": samples[0].task_id if samples else "?",
            "samples": [asdict(s) for s in samples],
        }
        data["tasks"].append(task_entry)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {path}")


def print_summary(bench: BenchmarkResult) -> None:
    print()
    print("=" * 60)
    print(f"HumanEval Benchmark Results")
    print(f"  Engine:    {bench.engine_type} ({bench.model_name})")
    print(f"  Remasking: {bench.remasking or 'n/a'}")
    print(f"  Mode:      {bench.mode}")
    print(f"  k:         {bench.k}")
    print(f"  Tasks:     {bench.n_tasks}")
    print(f"  pass@{bench.k}:   {bench.pass_at_k:.4f}  ({bench.pass_at_k * 100:.1f}%)")
    print(f"  Mean gen:  {bench.mean_generation_time_s:.2f}s")

    hits = misses = detections = 0
    for samples in bench.results:
        for s in samples:
            if s.speculative_hit is not None:
                detections += 1
                if s.speculative_hit:
                    hits += 1
                else:
                    misses += 1
    if detections:
        print(f"  Speculative: {hits} HIT / {misses} MISS / {detections} total")
        avg_saved = sum(
            s.time_saved for samples in bench.results for s in samples
            if s.time_saved is not None and s.time_saved > 0
        )
        if hits:
            print(f"  Avg time saved (HITs): {avg_saved / hits:.3f}s")

    print("=" * 60)
