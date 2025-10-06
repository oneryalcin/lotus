# LOTUS CLI Tools

Command-line interfaces for LOTUS semantic data processing.

## Quick Start

### Installation

```bash
# Install LOTUS
pip install -e .

# Set your API key
export OPENAI_API_KEY=your-key-here
```

### Basic Usage

```bash
# Filter data
python lotus_cli.py filter data.csv --condition "{text} is about AI"

# Rank results
python lotus_cli.py topk data.csv --criteria "Which {title} is most interesting?" --k 5

# Extract structured data
python lotus_cli.py extract people.csv --input-cols bio --fields "name,age,city"

# Search ArXiv
python lotus_cli.py search arxiv "deep learning" --num 10
```

## Tools

### 1. lotus_cli.py - Interactive CLI

Single-command interface for semantic operations.

**Commands:**
- `filter` - Filter rows by semantic condition
- `map` - Transform data with natural language
- `extract` - Extract structured fields from text
- `topk` - Rank and select top-k results
- `search` - Semantic web search

**Examples:**

```bash
# Filter CSV
python lotus_cli.py filter courses.csv \
  --condition "{Description} is about machine learning" \
  --output ml_courses.csv

# Map transformation
python lotus_cli.py map courses.csv \
  --instruction "What career uses {Course Name}?" \
  --suffix "_career" \
  --output enriched.csv

# Extract from unstructured text
python lotus_cli.py extract resumes.csv \
  --input-cols resume_text \
  --fields "name,email,skills,experience_years" \
  --output structured_resumes.csv

# Rank by semantic criteria
python lotus_cli.py topk papers.csv \
  --criteria "Which {abstract} describes the most novel approach?" \
  --k 10 \
  --output top_papers.csv

# Search and rank ArXiv
python lotus_cli.py search arxiv "semantic operators" \
  --num 20 \
  --rank "Which {abstract} is most relevant to data processing?" \
  --topk 5 \
  --output relevant_papers.csv

# Show token usage
python lotus_cli.py filter data.csv \
  --condition "{text} is high quality" \
  --show-usage
```

### 2. lotus_pipeline.py - Pipeline Runner

YAML-based pipeline executor for complex workflows.

**Features:**
- Multi-step processing
- Variable passing between steps
- Declarative configuration
- Dry-run validation

**Pipeline Example:**

```yaml
# pipeline.yaml
model:
  name: gpt-4o-mini

steps:
  # Step 1: Load and filter
  - type: filter
    name: ai_papers
    source: papers.csv
    condition: "{abstract} is about artificial intelligence"
    output: ai_papers.csv

  # Step 2: Extract information
  - type: extract
    name: paper_info
    input: ai_papers
    input_cols: [abstract, introduction]
    output_cols:
      methods: "What methods were used?"
      datasets: "What datasets were used?"
      results: "What were the key results?"
    output: paper_analysis.csv

  # Step 3: Rank papers
  - type: topk
    name: top_papers
    input: paper_info
    criteria: "Which {methods} is most innovative?"
    k: 5
    output: top_innovative_papers.csv

  # Step 4: Summarize
  - type: aggregate
    input: top_papers
    instruction: "Summarize the key trends in {methods} and {results}"
    output: summary.json
```

**Usage:**

```bash
# Run pipeline
python lotus_pipeline.py pipeline.yaml

# Validate without running
python lotus_pipeline.py pipeline.yaml --dry-run

# Show token usage
python lotus_pipeline.py pipeline.yaml --show-usage
```

## Supported Operations

### Filter
Filter rows based on natural language predicate.

```bash
lotus_cli.py filter INPUT --condition CONDITION [--output OUTPUT]
```

### Map
Transform each row with natural language instruction.

```bash
lotus_cli.py map INPUT --instruction INSTRUCTION [--suffix SUFFIX] [--output OUTPUT]
```

### Extract
Extract structured fields from unstructured text.

```bash
lotus_cli.py extract INPUT --input-cols COLS --fields FIELDS [--output OUTPUT]
```

Fields can be:
- Simple: `"name,age,city"` - extracts fields without descriptions
- JSON: `'{"name": "person's name", "age": "age in years"}'` - with descriptions

### TopK
Rank rows by semantic criteria and select top-k.

```bash
lotus_cli.py topk INPUT --criteria CRITERIA [--k K] [--output OUTPUT]
```

### Search
Search web sources and optionally rank results.

```bash
lotus_cli.py search CORPUS QUERY [--num NUM] [--rank CRITERIA] [--topk K] [--output OUTPUT]
```

Supported corpora:
- `arxiv` - ArXiv papers (no API key needed)
- `google` - Google Search (requires SERPAPI_API_KEY)
- `scholar` - Google Scholar (requires SERPAPI_API_KEY)
- `bing` - Bing Search (requires BING_API_KEY)
- `tavily` - Tavily Search (requires TAVILY_API_KEY)
- `you` - You.com Search (requires YOU_API_KEY)

## Pipeline Step Types

### filter
```yaml
- type: filter
  name: step_name
  source: input.csv  # or use 'input: previous_step'
  condition: "{column} matches some criteria"
  output: output.csv  # optional
```

### map
```yaml
- type: map
  name: step_name
  input: previous_step
  instruction: "Transform {column} somehow"
  suffix: "_new_column"
  output: output.csv
```

### extract
```yaml
- type: extract
  name: step_name
  input: previous_step
  input_cols: [col1, col2]
  output_cols:
    field1: "description of field1"
    field2: "description of field2"
  output: output.csv
```

### topk
```yaml
- type: topk
  name: step_name
  input: previous_step
  criteria: "Rank by {column}"
  k: 5
  output: output.csv
```

### join
```yaml
- type: join
  name: step_name
  input: left_step  # left dataframe
  right: right_step  # right dataframe
  condition: "{left_col:left} relates to {right_col:right}"
  output: output.csv
```

### search
```yaml
- type: search
  name: step_name
  corpus: arxiv
  query: "search query"
  num_results: 10
  output: output.csv
```

### aggregate
```yaml
- type: aggregate
  name: step_name
  input: previous_step
  instruction: "Summarize the {column} data"
  output: summary.json
```

## Configuration

### Model Selection

```bash
# Use different model
python lotus_cli.py filter data.csv --model gpt-4o --condition "..."

# In pipeline
model:
  name: gpt-4o
```

### Environment Variables

Required API keys:
- `OPENAI_API_KEY` - For OpenAI models (gpt-4o, gpt-4o-mini, etc.)
- `GEMINI_API_KEY` - For Google Gemini models
- `SERPAPI_API_KEY` - For Google/Scholar search
- `BING_API_KEY` - For Bing search
- `TAVILY_API_KEY` - For Tavily search
- `YOU_API_KEY` - For You.com search

## Tips and Best Practices

### 1. Start Small
Test on a small subset of data before running on full dataset.

```bash
# Test on first 10 rows
head -10 data.csv > sample.csv
python lotus_cli.py filter sample.csv --condition "..."
```

### 2. Use Show Usage
Monitor token usage to estimate costs.

```bash
python lotus_cli.py filter data.csv --condition "..." --show-usage
```

### 3. Chain with Unix Tools

```bash
# Filter -> Extract -> Select columns
python lotus_cli.py filter data.csv --condition "..." | \
  python lotus_cli.py extract - --input-cols text --fields "name,date" | \
  cut -d, -f1,2
```

### 4. Pipeline for Complex Workflows
Use pipelines for reproducible multi-step processing.

### 5. Version Control Your Pipelines
Store pipeline YAML in git for reproducibility.

### 6. Use Dry Run
Validate pipelines before executing.

```bash
python lotus_pipeline.py pipeline.yaml --dry-run
```

## Troubleshooting

### "No module named 'lotus'"
Install lotus: `pip install -e .`

### "Module not found" for web search
Install required extras:
```bash
pip install arxiv  # for ArXiv
pip install 'lotus-ai[serpapi]'  # for Google/Scholar
pip install 'lotus-ai[web_search]'  # for Bing/Tavily/You
```

### Rate Limiting
Add delays between operations or use caching:
```python
lotus.settings.configure(lm=lm, enable_cache=True)
```

### High Costs
- Use smaller models (gpt-4o-mini instead of gpt-4o)
- Test on samples first
- Use `--show-usage` to monitor tokens
- Enable caching for repeated queries

## Examples Gallery

### 1. Research Paper Analysis
```bash
# Search for papers
python lotus_cli.py search arxiv "large language models" --num 50 --output papers.csv

# Filter recent papers
python lotus_cli.py filter papers.csv \
  --condition "{published} is from 2024 or later" \
  --output recent.csv

# Extract key info
python lotus_cli.py extract recent.csv \
  --input-cols abstract \
  --fields "methods,datasets,results,limitations" \
  --output analysis.csv

# Rank by novelty
python lotus_cli.py topk analysis.csv \
  --criteria "Which {methods} is most innovative?" \
  --k 10 \
  --output top_papers.csv
```

### 2. Customer Feedback Analysis
```yaml
# feedback_analysis.yaml
model:
  name: gpt-4o-mini

steps:
  - type: filter
    name: english_feedback
    source: feedback.csv
    condition: "{comment} is written in English"

  - type: extract
    name: sentiment_analysis
    input: english_feedback
    input_cols: [comment]
    output_cols:
      sentiment: "positive, negative, or neutral"
      topic: "main topic of the comment"
      urgency: "high, medium, or low urgency"

  - type: topk
    name: urgent_issues
    input: sentiment_analysis
    criteria: "Which {comment} represents the most urgent customer issue?"
    k: 20
    output: urgent_feedback.csv
```

### 3. Content Moderation
```bash
# Flag inappropriate content
python lotus_cli.py filter comments.csv \
  --condition "{text} contains inappropriate or offensive content" \
  --output flagged.csv

# Extract reasons
python lotus_cli.py map flagged.csv \
  --instruction "Why is {text} inappropriate? Be specific." \
  --suffix "_reason" \
  --output moderation_report.csv
```

### 4. Data Quality Checks
```yaml
model:
  name: gpt-4o-mini

steps:
  - type: map
    name: email_validation
    source: contacts.csv
    instruction: "Is {email} a valid email address? Answer yes or no."
    suffix: "_valid"

  - type: filter
    name: valid_contacts
    input: email_validation
    condition: "{_valid} is yes"
    output: clean_contacts.csv
```

## Contributing

Have ideas for new CLI features? The codebase is simple and extensible:
- `lotus_cli.py` - Add new subcommands
- `lotus_pipeline.py` - Add new step types

## License

Same as LOTUS (MIT)
