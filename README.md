# Nabla

**Type-Aware HuggingFace Model Analyzer**

Nabla is a CLI tool that analyzes HuggingFace model configuration files to extract architectural details and resource requirements. It implements a structured pipeline to read `config.json`, `generation_config.json`, and other artifacts without downloading model weights.

## 🚀 Features

*   **Fact Extraction**: Extracts explicit config values using a strict schema.
*   **Model Routing**: Classification of model types (Text, VLM, Audio) based on config classes.
*   **Type-Specific Interpretation**: Adaptable analysis context based on the detected model type.
*   **Config Analysis**: Parses remote configuration files directly from HuggingFace.
*   **TUI**: Terminal user interface for model selection and interaction.
*   **History**: Locally caches recent model IDs.
*   **Clipboard**: Copy analysis prompts or results to clipboard.

## 📦 Installation

Requires Python 3.10+.

```bash
git clone https://github.com/yourusername/nabla.git
cd nabla
pip install -e .
```

## ⚡ Usage

### Quick Start (Interactive Mode)

Simply run the launcher for your OS:

**Windows:**
```pwsh
nabla.bat
```

**Linux / macOS:**
```bash
./nabla.sh
```

Or using Python directly:
```bash
python -m nabla
```

Follow the on-screen instructions to:
1.  Enter a HuggingFace Model ID (e.g., `microsoft/phi-2` or `Llama-3.2-11B-Vision-Instruct`).
2.  Select which config files to include in the analysis.
3.  Let the LLM analyze the architecture.

### CLI Arguments

You can also run Nabla with direct arguments:

```bash
# Analyze a specific model
python -m nabla --repo microsoft/phi-2

# Compare two models
python -m nabla --compare microsoft/phi-2 google/gemma-2b

# Force specific LLM provider (via LiteLLM)
python -m nabla --model openrouter/google/gemini-2.0-flash-exp
```

## ⚙️ Configuration

Nabla uses `LiteLLM` to talk to various providers (OpenAI, Anthropic, Google, OpenRouter, etc.).

By default, it uses `gpt-4o-mini`, but you can change this interactively or via CLI.

**Setting API Keys:**
Nabla looks for environment variables for API keys (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`).
You can also set them in a `.env` file or export them in your shell.

## 🛠️ Architecture

Nabla operates in 3 distinct steps to ensure accuracy:

1.  **Router (Step 0)**: Analyzes file structure and config classes to determine the *Model Type* (Text, VLM, Audio, etc.).
2.  **Extractor (Step 1)**: Extracts raw values into a strict JSON schema with provenance tracking (Source file + Path).
3.  **Interpreter (Step 2)**: Uses a type-specific System Prompt to interpret those facts into human-readable conclusions (e.g., "Good for RAG due to 128k context").

## 📄 License

MIT License
