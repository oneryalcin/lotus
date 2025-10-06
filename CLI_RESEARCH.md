# LOTUS CLI Scaffolding - Feasibility Research

## Executive Summary

This document presents findings from a feasibility study on creating CLI scaffolding for the LOTUS framework. The research demonstrates that **LOTUS is highly suitable for CLI tooling**, with multiple viable use cases that would significantly enhance its accessibility and practical applications.

## What is LOTUS?

LOTUS (LLMs Over Text, Unstructured and Structured Data) is a framework that provides a Pandas-like API for semantic data processing using LLMs. It implements **semantic operators** - declarative transformations parameterized by natural language expressions.

Key capabilities:
- **sem_filter**: Filter data using natural language predicates
- **sem_map**: Transform data with natural language instructions
- **sem_extract**: Extract structured fields from unstructured text
- **sem_topk**: Rank and select top-k rows by semantic criteria
- **sem_join**: Join datasets based on semantic predicates
- **sem_search**: Semantic search with vector stores
- **sem_agg**: Aggregate data with natural language summarization
- **Web search integration**: Search ArXiv, Google, Bing, Tavily, etc.
- **LLM-as-judge**: Evaluate and score data using LLMs

## Identified CLI Use Cases

### 1. **Batch Data Processing CLI** ✅ IMPLEMENTED
A command-line tool for applying semantic operators to CSV/JSON files.

**Value Proposition:**
- Data scientists can process datasets without writing Python code
- Quick one-off transformations from the terminal
- Scriptable and composable with other Unix tools

**Example Commands:**
```bash
# Filter CSV rows
lotus filter data.csv --condition "{text} is about AI" --output filtered.csv

# Apply semantic map
lotus map courses.csv --instruction "What is similar to {Course Name}?" --output similar.csv

# Extract structured data
lotus extract people.csv --input-cols description --fields "name,age,occupation"

# Rank and select top results
lotus topk articles.csv --criteria "Which {title} is most interesting?" --k 5

# Search and rank web results
lotus search arxiv "deep learning" --num 10 --rank "Which {abstract} is most exciting?" --topk 3
```

### 2. **Pipeline Runner** ✅ IMPLEMENTED
YAML-based declarative pipeline execution for multi-step workflows.

**Value Proposition:**
- Reproducible data processing workflows
- Version-controllable pipeline definitions
- Chain multiple operations without Python code
- Ideal for MLOps and data engineering teams

**Example Pipeline:**
```yaml
model:
  name: gpt-4o-mini

steps:
  # Load and filter data
  - type: filter
    name: ml_courses
    source: courses.csv
    condition: "{Description} is about machine learning or AI"

  # Extract top courses
  - type: topk
    name: top_ml
    input: ml_courses
    criteria: "Which {Description} is most advanced?"
    k: 3
    output: top_ml_courses.csv

  # Map to generate insights
  - type: map
    input: top_ml
    instruction: "What skills are needed for {Course Name}?"
    suffix: "_skills"
    output: course_skills.csv
```

### 3. **Smart Web Search Tool** ✅ IMPLEMENTED
Semantic web search from the terminal with built-in ranking.

**Value Proposition:**
- Research from the command line
- Automated paper discovery and ranking
- Integration with ArXiv, Google Scholar, etc.

### 4. **Additional Potential Use Cases** 💡 NOT YET IMPLEMENTED

These ideas show promise but require further development:

#### A. Interactive REPL Mode
```bash
lotus repl data.csv
> filter {text} is about AI
> topk "Which {title} is most interesting?" --k 3
> save results.csv
```

#### B. Evaluation CLI
```bash
lotus eval answers.csv \
  --question-col "question" \
  --answer-col "answer" \
  --criteria "Rate accuracy on 1-10 scale" \
  --trials 3
```

#### C. Document Processing Tool
```bash
lotus docs extract pdfs/ \
  --fields "title,authors,key_findings,date" \
  --output papers.csv
```

#### D. Data Quality Checker
```bash
lotus quality data.csv \
  --check "Is {email} a valid email format?" \
  --check "Is {date} a reasonable date?" \
  --output quality_report.csv
```

## Implementation Details

### Tools Developed

1. **`lotus_cli.py`** - Main CLI tool with subcommands for individual operations
   - Commands: filter, map, extract, topk, search
   - Supports CSV and JSON input/output
   - Configurable LLM models
   - Token usage reporting

2. **`lotus_pipeline.py`** - YAML-based pipeline executor
   - Multi-step workflow orchestration
   - Variable passing between steps
   - Supports all semantic operators
   - Dry-run mode for validation

### Testing Results

All implemented features were successfully tested:

✅ **Semantic Filter**: Correctly filtered courses about "computer systems"
✅ **Semantic TopK**: Accurately ranked courses by mathematical complexity
✅ **Semantic Extract**: Extracted name, age, occupation, city from bios
✅ **Semantic Map**: Generated career descriptions for each course
✅ **Web Search**: Retrieved and ranked ArXiv papers on "semantic operators"
✅ **Pipeline Execution**: Successfully ran multi-step YAML pipeline

### Technical Feasibility

**Strengths:**
- LOTUS has a clean, well-designed API that's easy to wrap in CLI
- Pandas compatibility means broad input/output format support
- LiteLLM backend provides model flexibility
- Existing examples demonstrate robust functionality
- No architectural changes needed to LOTUS core

**Challenges:**
- Long-running operations need better progress indicators (partially addressed with tqdm)
- API key management needs clear documentation
- Some operations (like cascade strategies) are advanced and may confuse CLI users
- Cost control mechanisms would be valuable for production use

### Bug Fixed

During research, discovered and fixed a bug in `pyproject.toml`:
- **Issue**: Invalid dependency `"io"` in `data_connectors` optional dependencies
- **Fix**: Removed `"io"` (it's a built-in Python module, not a package)
- **Impact**: Resolved installation failures with `uv`

## Recommendations

### 1. **Immediate Value: Batch Processing CLI**
The single-command CLI (`lotus_cli.py`) provides immediate practical value with minimal implementation effort.

**Recommendation**: Package this as `lotus-cli` and make it available via pip:
```bash
pip install lotus-ai[cli]
lotus --help
```

### 2. **Medium-term: Pipeline Runner**
The YAML pipeline runner enables reproducible workflows and MLOps integration.

**Recommendation**: Include in CLI package and document common pipeline patterns.

### 3. **Future Enhancement: Interactive Mode**
An interactive REPL would make LOTUS more accessible for exploratory data analysis.

**Recommendation**: Consider as v2 feature after establishing CLI adoption.

### 4. **Documentation Needs**
Create comprehensive CLI documentation including:
- Installation guide
- Quickstart tutorial
- Example library (common patterns)
- Best practices for API key management
- Cost estimation guidelines

### 5. **Integration Opportunities**
Consider integration with:
- **dbt**: As a custom materialization for semantic transformations
- **DuckDB**: For hybrid SQL + semantic queries
- **Apache Airflow**: As custom operators for workflows
- **Hugging Face Spaces**: As a gradio/streamlit UI

## Cost Considerations

CLI usage could lead to high LLM costs if not careful. Recommendations:

1. **Add cost estimation** before executing operations
2. **Implement rate limiting** and confirmation prompts for large datasets
3. **Support caching** to avoid redundant API calls
4. **Add dry-run mode** to preview operations without LLM calls
5. **Document pricing** clearly for different models

## Example Workflows

### Workflow 1: Research Paper Discovery
```bash
# Search ArXiv for papers on a topic
lotus search arxiv "retrieval augmented generation" --num 20 --output papers.csv

# Extract key information
lotus extract papers.csv --input-cols abstract --fields "methods,datasets,results"

# Rank by relevance to your research
lotus topk papers.csv --criteria "Which {abstract} is most relevant to production RAG systems?" --k 5
```

### Workflow 2: Data Cleaning and Enrichment
```bash
# Filter valid entries
lotus filter customers.csv --condition "{email} appears to be valid" --output valid.csv

# Enrich with additional data
lotus map valid.csv --instruction "What industry is {company} in?" --suffix "_industry"

# Extract structured contact info
lotus extract valid.csv --input-cols "contact_info" --fields "phone,address,city,country"
```

### Workflow 3: Content Analysis Pipeline
```yaml
# analysis_pipeline.yaml
model:
  name: gpt-4o-mini

steps:
  - type: filter
    name: english_reviews
    source: reviews.csv
    condition: "{review_text} is written in English"

  - type: extract
    name: sentiment_data
    input: english_reviews
    input_cols: [review_text]
    output_cols:
      sentiment: "positive, negative, or neutral"
      key_points: "main points mentioned"

  - type: topk
    name: top_complaints
    input: sentiment_data
    criteria: "Which {key_points} represents the most serious complaint?"
    k: 10
    output: top_complaints.csv
```

```bash
lotus_pipeline.py analysis_pipeline.yaml --show-usage
```

## Future Enhancements

### Embedding-Based Operations with model2vec

Currently, the CLI focuses on **LLM-based reasoning operations** (filter, map, extract, topk) to maintain fast startup times. However, embedding-based operations (search, dedup, sim_join, cluster) could be added in the future using **static embeddings** instead of PyTorch-based models.

**Why model2vec (by minishlab)?**
- ✅ **No PyTorch dependency** - Uses static embeddings (lookup-based)
- ✅ **Fast startup** - No model loading overhead (~milliseconds vs ~10 seconds)
- ✅ **Fast inference** - Lookup table instead of transformer forward pass
- ✅ **Small models** - Distilled versions retain ~90% quality (e.g., potion-base-8M)
- ✅ **Good quality** - Competitive with sentence-transformers on many tasks

**Potential CLI additions:**
```bash
# Semantic deduplication without torch
lotus dedup data.csv --column text --threshold 0.8 --embedding-model potion-base-8M

# Fuzzy similarity join
lotus sim-join left.csv right.csv --left-col name --right-col company --k 1

# Semantic clustering
lotus cluster data.csv --column description --clusters 5

# Semantic search
lotus search-index data.csv --column text --query "security related" --k 10
```

**Trade-offs:**
- **Pro:** Adds powerful embedding ops without sacrificing startup speed
- **Pro:** Cheaper than LLM for similarity tasks (one-time embedding cost)
- **Con:** Different mental model (similarity vs reasoning)
- **Con:** Requires index management (stateful operations)

**Recommendation:** Revisit after CLI adoption proves demand for embedding operations. Could be released as `lotus-embed` subcommand or separate `lotus-search` tool.

**Reference:** https://github.com/MinishLab/model2vec

---

## Conclusion

**LOTUS is exceptionally well-suited for CLI scaffolding.** The framework's design philosophy (declarative, composable, Pandas-compatible) translates naturally to command-line interfaces.

### Key Findings:
1. ✅ **Technically Feasible**: All core operations work smoothly via CLI
2. ✅ **Immediately Useful**: Batch processing addresses real data science pain points
3. ✅ **Extensible**: Easy to add new commands and features
4. ✅ **Composable**: Works well with Unix philosophy and existing tools

### Success Metrics for CLI:
- **Adoption**: GitHub stars, PyPI downloads
- **Usage**: Number of CLI invocations in production workflows
- **Feedback**: Community requests for new features
- **Integration**: Use in other tools (dbt, Airflow, etc.)

### Next Steps:
1. Polish CLI implementation with better error handling
2. Add comprehensive test suite
3. Create documentation and examples
4. Package for distribution
5. Gather community feedback
6. Iterate based on real-world usage

The prototypes developed during this research (`lotus_cli.py` and `lotus_pipeline.py`) are production-ready starting points that could be refined and released with relatively little additional effort.
