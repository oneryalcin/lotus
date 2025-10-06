# LOTUS CLI - Quick Start Guide

## What is this?

Two command-line tools that make LOTUS semantic data processing accessible from your terminal:

1. **`lotus_cli.py`** - Single commands for quick operations
2. **`lotus_pipeline.py`** - Multi-step workflows defined in YAML

## 5-Minute Demo

### Setup
```bash
# You're already in the lotus directory
# Make sure you have your API key
export OPENAI_API_KEY=your-key-here

# Install dependencies (already done)
# pip install -e .
```

### Example 1: Filter Data (10 seconds)
```bash
# Create sample data
cat > data.csv << 'EOF'
text
"Python is a programming language"
"I love hiking in the mountains"
"Machine learning is fascinating"
EOF

# Filter for tech-related content
python lotus_cli.py filter data.csv \
  --condition "{text} is about technology or programming"
```

**Output:**
```
text
Python is a programming language
Machine learning is fascinating
```

### Example 2: Extract Structured Data (20 seconds)
```bash
# Create sample bios
cat > people.csv << 'EOF'
bio
"Alice is a 25 year old engineer from NYC"
"Bob is a 30 year old designer from LA"
EOF

# Extract structured fields
python lotus_cli.py extract people.csv \
  --input-cols bio \
  --fields "name,age,occupation,city"
```

**Output:**
```
bio,name,age,occupation,city
"Alice is a 25 year old engineer from NYC",Alice,25,engineer,NYC
"Bob is a 30 year old designer from LA",Bob,30,designer,LA
```

### Example 3: Web Search & Rank (30 seconds)
```bash
# Search ArXiv and rank by relevance
python lotus_cli.py search arxiv "semantic operators" \
  --num 5 \
  --rank "Which {abstract} is most relevant to data processing?" \
  --topk 3
```

### Example 4: Multi-Step Pipeline (60 seconds)
```bash
# Create pipeline config
cat > my_pipeline.yaml << 'EOF'
model:
  name: gpt-4o-mini

steps:
  - type: search
    name: papers
    corpus: arxiv
    query: "large language models"
    num_results: 10

  - type: topk
    input: papers
    criteria: "Which {abstract} discusses practical applications?"
    k: 3
    output: top_papers.csv
EOF

# Run the pipeline
python lotus_pipeline.py my_pipeline.yaml
```

## All Commands

### lotus_cli.py

```bash
# Filter rows
lotus_cli.py filter INPUT --condition "..." [--output OUT]

# Transform data
lotus_cli.py map INPUT --instruction "..." [--output OUT]

# Extract fields
lotus_cli.py extract INPUT --input-cols COLS --fields "a,b,c" [--output OUT]

# Rank results
lotus_cli.py topk INPUT --criteria "..." --k 5 [--output OUT]

# Web search
lotus_cli.py search CORPUS QUERY [--rank "..."] [--topk K] [--output OUT]
```

### lotus_pipeline.py

```bash
# Run pipeline
lotus_pipeline.py pipeline.yaml

# Validate without running
lotus_pipeline.py pipeline.yaml --dry-run

# Show token usage
lotus_pipeline.py pipeline.yaml --show-usage
```

## Real-World Examples

### Research Literature Review
```yaml
# research.yaml
model:
  name: gpt-4o-mini

steps:
  - type: search
    corpus: arxiv
    query: "your research topic"
    num_results: 50
    name: papers

  - type: filter
    input: papers
    condition: "{abstract} discusses practical applications"
    name: relevant

  - type: extract
    input: relevant
    input_cols: [abstract]
    output_cols:
      methods: null
      datasets: null
      results: null
    name: analyzed

  - type: topk
    input: analyzed
    criteria: "Most innovative {methods}"
    k: 10
    output: top_papers.csv
```

### Data Cleaning
```bash
# Validate emails
python lotus_cli.py filter customers.csv \
  --condition "{email} is a valid email format" \
  --output clean.csv

# Enrich with industry
python lotus_cli.py map clean.csv \
  --instruction "What industry is {company} in?" \
  --suffix "_industry" \
  --output enriched.csv
```

### Content Analysis
```bash
# Extract sentiment
python lotus_cli.py extract reviews.csv \
  --input-cols review_text \
  --fields "sentiment,key_points,rating_implied" \
  --output analyzed.csv

# Find top issues
python lotus_cli.py topk analyzed.csv \
  --criteria "Which {key_points} is most urgent?" \
  --k 10 \
  --output urgent_issues.csv
```

## Tips

1. **Start small** - Test on 5-10 rows first
2. **Use --show-usage** - Monitor token costs
3. **Dry-run pipelines** - Validate before executing
4. **Save intermediates** - Add outputs at key steps
5. **Version control** - Store pipelines in git

## Common Issues

**"Module not found"**
→ Run: `pip install -e .`

**"API key not set"**
→ Export: `export OPENAI_API_KEY=...`

**"No module for web search"**
→ Install: `pip install arxiv` (for ArXiv)

**"Too expensive"**
→ Use: `--model gpt-4o-mini` or test on samples first

## What's Next?

See detailed documentation:
- `CLI_README.md` - Complete reference
- `CLI_RESEARCH.md` - Feasibility study
- `CLI_SUMMARY.md` - Project overview
- `examples/cli_examples/` - More examples

## Support

- LOTUS Docs: https://lotus-ai.readthedocs.io/
- LOTUS Slack: https://join.slack.com/t/lotus-fnm8919/shared_invite/...
- GitHub Issues: https://github.com/lotus-data/lotus/issues

---

**Built with LOTUS** - Semantic operators for the command line 🌸
