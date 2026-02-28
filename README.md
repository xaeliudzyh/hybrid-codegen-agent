# hybrid-codegen-agent

Agent-based code generation system with **autoregressive** and **diffusion** language models.
The key research feature is **speculative execution** - detecting function calls in intermediate diffusion states and executing them in the background before generation completes.

## How it works

```
Standard:     [===== generate =====] → [parse] → [== execute ==]  → done
Speculative:  [===== generate =====] → [parse] → [result ready!]  → done
                        ↑ detected early          ↑ executed in background
```

The diffusion model (LLaDA) generates all tokens in parallel and refines them over N steps.
At each step, `EarlyFunctionDetector` decodes the partial sequence and looks for `<function_call>` tags.
Once found, `SpeculativeExecutor` runs the function in a background thread.
When generation finishes, the agent compares the final function call with the speculative one:
**HIT** → use the cached result; **MISS** → discard and re-execute.

## Project Structure

```
src/
├── agent/              CodeGenAgent - orchestration, HIT/MISS logic, metrics
├── engines/            GenerativeEngine (ABC), AutoregressiveEngine (LLaMA)
├── diffusion/          DiffusionEngine (LLaDA), EarlyFunctionDetector, SpeculativeExecutor
└── function_calling/   FunctionCall parsing, FunctionRegistry, execute_code()
scripts/
└── run_agent.py        CLI entry point
tests/                  45 unit tests
```

## Installation

```bash
git clone https://github.com/your-repo/hybrid-codegen-agent.git
cd hybrid-codegen-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Autoregressive (baseline)

```bash
python scripts/run_agent.py \
    --engine autoregressive \
    --model unsloth/llama-2-7b-chat \
    --task "Write a fibonacci function"
```

### Diffusion (standard)

```bash
python scripts/run_agent.py \
    --engine diffusion \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --device cuda \
    --task "Write a fibonacci function"
```

### Diffusion with speculative execution

```bash
python scripts/run_agent.py \
    --engine diffusion \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --device cuda \
    --with-early-detection \
    --task "Write a fibonacci function"
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | *factorial example* | Task description (text or path to `.txt` file) |
| `--engine` | `autoregressive` | `autoregressive` or `diffusion` |
| `--model` | `unsloth/llama-2-7b-chat` | HuggingFace model name or local path |
| `--device` | `auto` | `cuda`, `cuda:0`, `cpu`, or `auto` |
| `--use-stub` | off | Use stub engine (no GPU needed, for testing) |
| `--max_iterations` | `5` | Max generate → execute cycles |
| **Speculative execution** |||
| `--with-early-detection` | off | Enable speculative execution (diffusion only) |
| `--min_step_ratio` | `0.1` | Skip first N% of steps (too noisy to decode) |
| `--check_interval` | `1` | Check every N-th step (higher = less overhead) |
| `--no-require-valid-json` | off | Accept any `<function_call>` match without JSON validation |
 

## Architecture

| Component | Role |
|-----------|------|
| **CodeGenAgent** | Orchestration: `run()` and `run_with_speculative_execution()` |
| **GenerativeEngine** | Abstract interface for text generation |
| **AutoregressiveEngine** | LLaMA-based baseline (HuggingFace Transformers) |
| **DiffusionEngine** | LLaDA-based engine with `generate_with_speculative_execution()` |
| **EarlyFunctionDetector** | `step_callback` - detects FC in intermediate diffusion states |
| **SpeculativeExecutor** | Runs detected function in `ThreadPoolExecutor` during generation |
| **FunctionRegistry** | Registration and safe execution of functions (`execute_code`) |

## HuggingFace Authentication

For gated models (e.g., `meta-llama/*`):

```bash
pip install huggingface_hub
huggingface-cli login
```

## License

MIT
