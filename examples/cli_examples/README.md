# LOTUS CLI Examples

This directory contains example pipelines demonstrating real-world uses of the LOTUS CLI tools.

## Examples

### 1. research_pipeline.yaml
**Academic Research Pipeline**

Searches ArXiv, filters papers, extracts key information, and generates a research summary.

```bash
cd /path/to/lotus
python lotus_pipeline.py examples/cli_examples/research_pipeline.yaml
```

**Use case**: Literature review and research synthesis

### 2. data_cleaning_pipeline.yaml
**Data Cleaning and Enrichment**

Validates customer data, extracts company information, and identifies high-value prospects.

**Prerequisites**: Requires a `customers.csv` file with columns:
- email
- company_description
- address

```bash
python lotus_pipeline.py examples/cli_examples/data_cleaning_pipeline.yaml
```

**Use case**: CRM data cleaning and lead qualification

## Running Examples

### Method 1: Direct Execution
```bash
python lotus_pipeline.py examples/cli_examples/research_pipeline.yaml
```

### Method 2: Dry Run First
```bash
# Validate pipeline without executing
python lotus_pipeline.py examples/cli_examples/research_pipeline.yaml --dry-run

# Then run it
python lotus_pipeline.py examples/cli_examples/research_pipeline.yaml
```

### Method 3: With Usage Tracking
```bash
python lotus_pipeline.py examples/cli_examples/research_pipeline.yaml --show-usage
```

## Creating Your Own Pipelines

### Basic Template

```yaml
model:
  name: gpt-4o-mini  # or gpt-4o, claude-3, etc.

steps:
  # Step 1: Load or search for data
  - type: search  # or use 'source: file.csv'
    name: my_data
    corpus: arxiv
    query: "your search query"
    num_results: 20

  # Step 2: Filter
  - type: filter
    name: filtered_data
    input: my_data
    condition: "{column} meets some criteria"

  # Step 3: Extract structured info
  - type: extract
    name: extracted
    input: filtered_data
    input_cols: [text_column]
    output_cols:
      field1: "description of field1"
      field2: null  # auto-infer from field name

  # Step 4: Rank
  - type: topk
    name: top_results
    input: extracted
    criteria: "Ranking criteria for {column}"
    k: 10
    output: final_output.csv
```

## Available Step Types

- **search**: Web search (arxiv, google, scholar, bing, tavily, you)
- **filter**: Keep rows matching condition
- **map**: Transform each row
- **extract**: Extract structured fields
- **topk**: Rank and select top-k
- **join**: Semantic join two datasets
- **aggregate**: Summarize entire dataset

See `CLI_README.md` in the root directory for complete documentation.

## Tips

1. **Start with small datasets** - Test with `num_results: 5` first
2. **Use meaningful step names** - They become variable names
3. **Add outputs at key stages** - Saves intermediate results
4. **Version control your pipelines** - They're reproducible workflows
5. **Monitor costs** - Use `--show-usage` flag

## Common Patterns

### Pattern 1: Search → Filter → Extract → Rank
For research and information gathering.

### Pattern 2: Load → Validate → Clean → Enrich
For data quality and enrichment.

### Pattern 3: Load → Filter → Join → Aggregate
For complex analytical workflows.

### Pattern 4: Multiple sources → Join → Analyze
For multi-source data integration.

## Need Help?

See the main LOTUS documentation and CLI_README.md for more details.
