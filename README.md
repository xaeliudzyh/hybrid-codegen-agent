# hybrid-codegen-agent

A research agent-based code generation system supporting both autoregressive and diffusion-based language models. The main focus is studying function calling mechanisms in diffusion language models and their impact on system latency.

## Project Structure

```
hybrid-codegen-agent/
├── src/
│   ├── agent/                 # Agent orchestration layer
│   ├── engines/               # Generative engines (autoregressive, diffusion)
│   ├── function_calling/      # Function call parsing, registry, execution
│   ├── diffusion/             # Diffusion-specific logic (placeholder)
│   └── experiments/           # Experiment runners
├── scripts/
│   └── run_agent.py           # CLI entry point
└── tests/                     # Unit tests
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/hybrid-codegen-agent.git
cd hybrid-codegen-agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running with Stub (for testing)

```bash
python scripts/run_agent.py --task "Write a fibonacci function" # or --task tasks/fibonacci_task.txt
```

### Running with Real Model (requires GPU)

```bash
python scripts/run_agent.py \
    --task "Write a fibonacci function in Python" \ 
    --model unsloth/llama-2-7b-chat \
    --no-use-stub \
    --device cuda
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | "Write a Python function..." | The code generation task |
| `--engine` | `autoregressive` | Engine type: `autoregressive` or `diffusion` |
| `--model` | `meta-llama/Llama-2-7b-hf` | HuggingFace model name or local path |
| `--use-stub` | (flag) | Use stub implementation for testing |
| `--no-use-stub` | (flag) | Use real model instead of stub |
| `--device` | `auto` | Device: `cuda`, `cuda:0`, `cpu`, or `auto` |

### Recommended Open Models

| Model | Size | Notes |
|-------|------|-------|
| `unsloth/llama-2-7b-chat` | ~14 GB | Open, no license required |
| `Qwen/Qwen2.5-7B-Instruct` | ~14 GB | High quality, open |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | ~2 GB | Fast, for quick tests |

### Using Specific GPU

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_agent.py \
    --task "Write a sort function" \ # or --task tasks/fibonacci_task.txt
    --model unsloth/llama-2-7b-chat \
    --no-use-stub
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## HuggingFace Authentication

For gated models (e.g., `meta-llama/*`), you need to:

1. Accept the license at https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
2. Login via CLI:
   ```bash
   pip install huggingface_hub
   huggingface-cli login #or hf auth login
   ```

## Architecture

- **CodeGenAgent**: Orchestration layer, model-independent
- **GenerativeEngine**: Abstract interface for text generation
- **AutoregressiveEngine**: LLaMA-based baseline implementation
- **DiffusionEngine**: LLaDA-based research implementation (in progress)
- **FunctionRegistry**: Explicit function registration and execution
- **FunctionCall**: Structured representation of function calls

## License

MIT
