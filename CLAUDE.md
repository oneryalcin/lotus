# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LOTUS (LLMs Over Text, Unstructured and Structured Data) is a framework for LLM-powered data processing with a Pandas-like API. It implements semantic operators that extend relational operations to unstructured data using AI.

This is a fork of lotus-data/lotus with:
- CLI tools built with Typer
- model2vec support for embeddings (no PyTorch dependency)
- vicinity vector store integration

## Development Commands

### Setup
```bash
# Create environment
conda create -n lotus python=3.10 -y
conda activate lotus

# Install in development mode
pip install -e .

# Install dev dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Testing
```bash
# Set required environment variables
export ENABLE_OPENAI_TESTS="true"
export ENABLE_LOCAL_TESTS="true"
export OPENAI_API_KEY="<your-key>"

# Run all tests
pytest

# Run specific test file
pytest tests/test_lm.py

# Run in parallel
pytest -n auto

# CI/CD tests are in .github/tests/
# Additional tests are in tests/
```

### Code Quality
```bash
# Run all pre-commit checks
pre-commit run --all-files

# Run specific hook
pre-commit run ruff --all-files

# Type checking
mypy lotus/
```

### CLI Tools
```bash
# Install as global CLI tool (via uv)
uv tool install .

# Run CLI commands
lotus filter data.csv --condition "{text} is about AI"
lotus-pipeline pipeline.yaml

# Or run directly in dev
python lotus_cli.py filter data.csv --condition "{text} is about AI"
python lotus_pipeline.py pipeline.yaml --dry-run
```

### Running Single Tests
```bash
# Run a specific test function
pytest tests/test_lm.py::test_model_initialization -v

# Run tests matching a pattern
pytest -k "filter" -v
```

## Architecture

### Core Components

**Semantic Operators** (`lotus/sem_ops/`):
- Each operator (sem_filter, sem_map, sem_extract, etc.) is in its own file
- Operators extend pandas DataFrame with semantic capabilities via monkey-patching
- Key operators: filter, map, extract, join, topk, agg, search, sim_join, dedup, cluster_by
- Operators are parameterized by natural language expressions (langex) with column refs in brackets: `{column_name}`

**Models** (`lotus/models/`):
- `LM` - Language models via LiteLLM (supports OpenAI, Ollama, vLLM, etc.)
- `RM` - Retrieval/embedding models (Model2VecRM, SentenceTransformersRM, LiteLLMRM)
- `Reranker` - CrossEncoder rerankers
- `lm.py` has extensive usage tracking and caching logic

**Settings** (`lotus/settings.py`):
- Global singleton for model configuration
- Configure with: `lotus.settings.configure(lm=lm, rm=rm, vs=vs)`
- Not thread-safe

**Vector Store** (`lotus/vector_store/`):
- `VicinityVS` - Default vector store using vicinity library
- Supports FAISS-based indexing for semantic search

**CLI** (`lotus_cli.py`, `lotus_pipeline.py`):
- `lotus_cli.py`: Single-command interface using Typer
- `lotus_pipeline.py`: YAML-based multi-step pipeline runner
- Default models:
  - LM: `gemini/gemini-2.5-flash-lite-preview-09-2025`
  - Embeddings: `minishlab/potion-base-8M` (model2vec)

### Key Patterns

**DataFrame Extension Pattern**:
Semantic operators are added to pandas DataFrame objects dynamically. When you import lotus, it monkey-patches pandas DataFrames to add methods like `sem_filter()`, `sem_map()`, etc.

**Natural Language Expressions (langex)**:
All semantic operators use natural language with column placeholders:
- `"{abstract} is about artificial intelligence"` (filter)
- `"Summarize {text}"` (map)
- `"Taking {Course Name} will help me learn {Skill}"` (join)

**Model Configuration Flow**:
1. Create model: `lm = LM(model="gpt-4o")`
2. Configure globally: `lotus.settings.configure(lm=lm)`
3. Operators automatically use configured models

**Caching**:
LM class supports caching to reduce API calls. Enable with `lotus.settings.configure(enable_cache=True)`.

**Usage Tracking**:
LM class tracks token usage. Access with `lm.print_total_usage()` or `lm.get_total_usage()`.

## Model Support

**LM Models** (via LiteLLM):
- OpenAI: `gpt-4o`, `gpt-4o-mini`
- Gemini: `gemini/gemini-2.5-flash-lite-preview-09-2025`
- Ollama: `ollama/llama3.2`
- vLLM: Set `api_base` parameter
- Any LiteLLM-supported model

**Embedding Models**:
- **Recommended**: Model2VecRM with models like `minishlab/potion-base-8M` (no PyTorch)
- SentenceTransformersRM (deprecated): Any HuggingFace model
- LiteLLMRM: Any LiteLLM embedding model

**Rerankers**:
- CrossEncoder models from SentenceTransformers

## CLI Architecture

### lotus_cli.py
- Built with Typer framework for type-safe CLI
- Each command corresponds to a semantic operator
- Commands: filter, map, extract, topk, search, dedup, cluster, index, semsearch, sim-join
- Supports CSV and JSON input/output
- Entry point: `lotus = "lotus_cli:main"`

### lotus_pipeline.py
- YAML-based pipeline executor
- Supports step dependencies via `input: step_name` or `source: file.csv`
- Pipeline structure:
  ```yaml
  model:
    name: gpt-4o-mini
    embedding_model: minishlab/potion-base-8M  # optional

  steps:
    - type: filter
      name: step_name
      source: input.csv
      condition: "{column} matches criteria"
      output: output.csv
  ```

## Testing Guidelines

**Test Locations**:
- `.github/tests/`: Core CI/CD tests for essential functionality
- `tests/`: Additional tests for non-core features and integrations

**Writing Tests**:
- Avoid assertions on exact model output (models are non-deterministic)
- Test that expected columns exist and contain non-empty values
- Mock external dependencies where possible
- Use descriptive test names

**Test Environment Variables**:
Required for tests that call models:
- `ENABLE_OPENAI_TESTS="true"`
- `ENABLE_LOCAL_TESTS="true"`
- `OPENAI_API_KEY="<key>"`

## Code Style

**Linting**: Ruff (see `.pre-commit-config.yaml`)
**Type Checking**: mypy with strict mode (see `mypy.ini`)
**Formatting**: Ruff format
**Line Length**: 120 chars

Pre-commit hooks run automatically on commit. Run manually with `pre-commit run --all-files`.

## Common Gotchas

**FAISS Threading**: Set `OMP_NUM_THREADS=1` to avoid threading issues on some platforms (see cluster command in CLI).

**Mac Installation**: Install FAISS via conda (`conda install -c pytorch faiss-cpu=1.8.0`) before pip installing lotus.

**Index Management**: Embedding-based operations (dedup, cluster, semsearch) require indices. CLI creates temp indices by default; use `--keep-index` and `--index-dir` to persist.

**Model API Keys**: Export appropriate API key for your model:
- `OPENAI_API_KEY` for OpenAI
- `GEMINI_API_KEY` for Gemini
- Search APIs: `SERPAPI_API_KEY`, `BING_API_KEY`, `TAVILY_API_KEY`, `YOU_API_KEY`

## Important Files

- `lotus/__init__.py` - Package entry point, imports all operators
- `lotus/models/lm.py` - Core LM class with caching and usage tracking
- `lotus/sem_ops/` - Individual semantic operator implementations
- `pyproject.toml` - Package config, dependencies, CLI entry points
- `examples/` - Usage examples for all operators
- `CLI_README.md` - Comprehensive CLI documentation
