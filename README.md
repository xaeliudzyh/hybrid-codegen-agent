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
tests/                  68 unit tests
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

### Diffusion with fc_priority remasking

```bash
python scripts/run_agent.py \
    --engine diffusion \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --device cuda \
    --remasking fc_priority \
    --fc_boost 0.3 \
    --with-early-detection \
    --task "Write a fibonacci function"
```

### Diffusion with structural_boost remasking

```bash
python scripts/run_agent.py \
    --engine diffusion \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --device cuda \
    --remasking structural_boost \
    --structural_boost 0.5 \
    --structural_window 3 \
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
| `--max_tokens` | `512` | Max tokens to generate per call (prompt + max_tokens ≤ 4096) |
| `--use-stub` | off | Use stub engine (no GPU needed, for testing) |
| `--max_iterations` | `5` | Max generate → execute cycles |
| **Diffusion parameters** |||
| `--steps` | `64` | Number of diffusion denoising steps |
| `--block_length` | `32` | Block size for semi-autoregressive generation |
| `--remasking` | `low_confidence` | Remasking strategy: `low_confidence`, `random`, `fc_priority`, `structural_boost` |
| `--fc_boost` | `0.2` | Additive confidence bonus for FC-anchor tokens (`fc_priority` strategy) |
| `--structural_boost` | `0.3` | Additive proximity bonus near fixed FC clusters (`structural_boost` strategy) |
| `--structural_window` | `5` | Half-window size for proximity detection (`structural_boost` strategy) |
| **Speculative execution** |||
| `--with-early-detection` | off | Enable speculative execution (diffusion only) |
| `--min_step_ratio` | `0.1` | Skip first N% of steps (too noisy to decode) |
| `--check_interval` | `1` | Check every N-th step (higher = less overhead) |
| `--no-require-valid-json` | off | Accept any `<function_call>` match without JSON validation |

### Remasking Strategies

| Strategy | Description |
|----------|-------------|
| `low_confidence` | Fix tokens with highest model confidence $p(\hat{x}_j)$ first. Default, best general quality. |
| `random` | Random selection — baseline for ablation studies. |
| `fc_priority` | `low_confidence` + additive boost for FC-structural tokens (`<function_call>`, `{`, `"name"`, etc.). Accelerates FC pattern formation for earlier speculative detection. |
| `structural_boost` | `low_confidence` + proximity bonus near already-fixed FC-anchor clusters (sliding window). Creates cascading crystallization — fixed FC tokens help unmask their neighbors faster. |
 

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
