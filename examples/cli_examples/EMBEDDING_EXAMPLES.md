# CLI Embedding Operations Examples

This directory contains examples for using LOTUS embedding operations from the command line. These operations use **model2vec** for fast, lightweight embeddings without PyTorch.

## Quick Start

All embedding operations require model2vec:

```bash
pip install 'lotus-ai[model2vec]'
```

## Available Commands

### 1. `lotus dedup` - Semantic Deduplication

Remove semantically similar duplicates from your data.

**Command line:**
```bash
lotus dedup courses.csv \
  --column "Course Name" \
  --threshold 0.65 \
  --output unique_courses.csv
```

**Pipeline:** See [embedding_dedup_example.yaml](embedding_dedup_example.yaml)

**Use cases:**
- Remove duplicate product listings
- Clean up course catalogs
- Deduplicate customer feedback

### 2. `lotus cluster` - Semantic Clustering

Group data into semantic clusters automatically.

**Command line:**
```bash
lotus cluster courses.csv \
  --column "Course Name" \
  --num-clusters 3 \
  --output categorized.csv
```

**Pipeline:** See [embedding_cluster_example.yaml](embedding_cluster_example.yaml)

**Use cases:**
- Automatically categorize documents
- Group similar support tickets
- Organize product catalogs

### 3. `lotus index` + `lotus semsearch` - Semantic Search

Create searchable indexes and query them with natural language.

**Command line:**
```bash
# Create index (one-time)
lotus index docs.csv \
  --column Content \
  --index-dir ./docs_index

# Search the index (reusable)
lotus semsearch \
  --data docs.csv \
  --index-dir ./docs_index \
  --column Content \
  --query "machine learning tutorials" \
  --k 5
```

**Pipeline:** See [embedding_search_example.yaml](embedding_search_example.yaml)

**Use cases:**
- Build a document search engine
- Find relevant knowledge base articles
- Query large datasets semantically

### 4. `lotus sim-join` - Fuzzy Similarity Joins

Join datasets with fuzzy name matching.

**Command line:**
```bash
lotus sim-join \
  --left jobs.csv \
  --right companies.csv \
  --left-col "Company Name" \
  --right-col "Company" \
  --k 1 \
  --output matched_jobs.csv
```

**Pipeline:** See [embedding_fuzzy_join_example.yaml](embedding_fuzzy_join_example.yaml)

**Use cases:**
- Match company name variations
- Link products across databases
- Entity resolution

## Pipeline Examples

All examples can be run as pipelines:

```bash
lotus-pipeline embedding_dedup_example.yaml
lotus-pipeline embedding_cluster_example.yaml
lotus-pipeline embedding_search_example.yaml
lotus-pipeline embedding_fuzzy_join_example.yaml
```

## Configuration Options

### Embedding Models

Default: `minishlab/potion-base-8M` (recommended)

Other options:
- `minishlab/potion-base-4M` - Smaller, faster
- `minishlab/potion-base-32M` - Higher quality, slower

Specify in pipeline:
```yaml
model:
  embedding_model: minishlab/potion-base-32M
```

Or command line:
```bash
lotus dedup data.csv --column Text --embedding-model minishlab/potion-base-32M
```

### Thresholds

For `dedup`:
- `0.5` - Aggressive (removes more)
- `0.65` - Balanced (recommended)
- `0.8` - Conservative (removes less)

Example:
```bash
lotus dedup data.csv --column Text --threshold 0.5
```

### Number of Results

For `semsearch`:
```bash
lotus semsearch --data docs.csv --index-dir ./idx --column Text --query "AI" --k 10
```

For `sim-join`:
```bash
lotus sim-join --left a.csv --right b.csv --left-col Name --right-col Title --k 3
```

## Performance

**model2vec benefits:**
- ✅ No PyTorch required
- ✅ Fast startup (milliseconds)
- ✅ Small models (8M-32M parameters)
- ✅ Good quality (~90% of transformer performance)

**Typical speeds:**
- Indexing: ~1000 docs/sec
- Search: <100ms for 10K docs
- Dedup: ~500 comparisons/sec

## Tips

1. **Reuse indexes:** Save index directories to avoid re-embedding
2. **Tune thresholds:** Start with 0.65 for dedup, adjust based on results
3. **Batch operations:** Use pipelines for multi-step workflows
4. **Check results:** Use `--output` to save and inspect results

## Troubleshooting

**Slow clustering?**
- Ensure you're using the latest version (includes OMP fix)
- Try smaller num_clusters first

**Index not found?**
- Check that index step comes before dedup/cluster/semsearch
- Verify index_dir path exists and is accessible

**Low quality matches?**
- Try a larger embedding model (32M instead of 8M)
- Adjust threshold (lower for more matches, higher for precision)

## More Examples

See also:
- [README.md](README.md) - General CLI examples
- [data_cleaning_pipeline.yaml](data_cleaning_pipeline.yaml) - LLM-based operations
- [research_pipeline.yaml](research_pipeline.yaml) - Multi-step research workflow
