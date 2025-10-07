#!/usr/bin/env python
"""
Lotus CLI - A command-line interface for semantic data processing

This CLI provides easy access to LOTUS semantic operators for batch data processing.
"""
import argparse
import json
import sys
from pathlib import Path


def setup_lotus(model: str = "gpt-4o-mini"):
    """Initialize LOTUS with the specified model."""
    import lotus
    from lotus.models import LM

    lm = LM(model=model)
    lotus.settings.configure(lm=lm)
    return lm


def setup_lotus_embeddings(embedding_model: str = "minishlab/potion-base-8M", lm_model: str | None = None):
    """Initialize LOTUS with embeddings support (model2vec + vicinity)."""
    import lotus
    from lotus.models import Model2VecRM
    from lotus.vector_store import VicinityVS

    rm = Model2VecRM(model=embedding_model)
    vs = VicinityVS(backend="BASIC", metric="cosine")

    if lm_model:
        from lotus.models import LM
        lm = LM(model=lm_model)
        lotus.settings.configure(lm=lm, rm=rm, vs=vs)
        return lm, rm, vs
    else:
        lotus.settings.configure(rm=rm, vs=vs)
        return None, rm, vs


def load_data(input_path: str):
    """Load data from CSV or JSON file."""
    import pandas as pd

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if path.suffix == '.csv':
        return pd.read_csv(input_path)
    elif path.suffix == '.json':
        return pd.read_json(input_path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .csv or .json")


def save_data(df, output_path: str):
    """Save data to CSV or JSON file."""
    path = Path(output_path)
    if path.suffix == '.csv':
        df.to_csv(output_path, index=False)
    elif path.suffix == '.json':
        df.to_json(output_path, orient='records', indent=2)
    else:
        raise ValueError(f"Unsupported output format: {path.suffix}. Use .csv or .json")


def cmd_filter(args):
    """Apply semantic filter to data."""
    lm = setup_lotus(args.model)
    df = load_data(args.input)

    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Applying filter: {args.condition}")

    result = df.sem_filter(args.condition)

    print(f"Filtered to {len(result)} rows")

    if args.output:
        save_data(result, args.output)
        print(f"Saved to {args.output}")
    else:
        print(result)

    if args.show_usage:
        lm.print_total_usage()


def cmd_map(args):
    """Apply semantic map to data."""
    lm = setup_lotus(args.model)
    df = load_data(args.input)

    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Applying map: {args.instruction}")

    result = df.sem_map(args.instruction, suffix=args.suffix)

    if args.output:
        save_data(result, args.output)
        print(f"Saved to {args.output}")
    else:
        print(result)

    if args.show_usage:
        lm.print_total_usage()


def cmd_extract(args):
    """Extract structured data from text."""
    lm = setup_lotus(args.model)
    df = load_data(args.input)

    print(f"Loaded {len(df)} rows from {args.input}")

    # Parse output columns from JSON or simple format
    if args.fields.startswith('{'):
        output_cols = json.loads(args.fields)
    else:
        # Simple format: "name,age,location" -> {"name": None, "age": None, "location": None}
        output_cols = {field.strip(): None for field in args.fields.split(',')}

    print(f"Extracting fields: {list(output_cols.keys())}")

    result = df.sem_extract(args.input_cols.split(','), output_cols)

    if args.output:
        save_data(result, args.output)
        print(f"Saved to {args.output}")
    else:
        print(result)

    if args.show_usage:
        lm.print_total_usage()


def cmd_topk(args):
    """Rank and select top-k rows."""
    lm = setup_lotus(args.model)
    df = load_data(args.input)

    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Ranking by: {args.criteria}")
    print(f"Selecting top {args.k} rows")

    result = df.sem_topk(args.criteria, K=args.k)

    if args.output:
        save_data(result, args.output)
        print(f"Saved to {args.output}")
    else:
        print(result)

    if args.show_usage:
        lm.print_total_usage()


def cmd_search(args):
    """Search web and apply semantic operations."""
    from lotus import WebSearchCorpus, web_search

    lm = setup_lotus(args.model)

    # Map corpus string to enum
    corpus_map = {
        'arxiv': WebSearchCorpus.ARXIV,
        'google': WebSearchCorpus.GOOGLE,
        'scholar': WebSearchCorpus.GOOGLE_SCHOLAR,
        'you': WebSearchCorpus.YOU,
        'bing': WebSearchCorpus.BING,
        'tavily': WebSearchCorpus.TAVILY,
    }

    corpus = corpus_map.get(args.corpus.lower())
    if not corpus:
        raise ValueError(f"Unknown corpus: {args.corpus}. Choose from: {list(corpus_map.keys())}")

    print(f"Searching {args.corpus} for: {args.query}")
    print(f"Retrieving {args.num} results")

    df = web_search(corpus, args.query, args.num)

    # Apply optional topk ranking
    if args.rank:
        print(f"Ranking by: {args.rank}")
        df = df.sem_topk(args.rank, K=args.topk if args.topk else args.num)

    if args.output:
        save_data(df, args.output)
        print(f"Saved to {args.output}")
    else:
        print(df)

    if args.show_usage:
        lm.print_total_usage()


def cmd_dedup(args):
    """Remove semantic duplicates from data."""
    _, rm, vs = setup_lotus_embeddings(args.embedding_model)
    df = load_data(args.input)

    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Deduplicating column: {args.column}")
    print(f"Threshold: {args.threshold}")

    # Create index directory name
    import tempfile
    import os
    index_dir = args.index_dir if args.index_dir else os.path.join(tempfile.gettempdir(), f"lotus_dedup_{os.getpid()}")

    # Index and deduplicate
    df = df.sem_index(args.column, index_dir)
    result = df.sem_dedup(args.column, threshold=args.threshold)

    removed = len(df) - len(result)
    print(f"Removed {removed} duplicates ({removed/len(df)*100:.1f}%)")
    print(f"Remaining: {len(result)} rows")

    if args.output:
        save_data(result, args.output)
        print(f"Saved to {args.output}")
    else:
        print(result)

    # Clean up temp index if not specified
    if not args.index_dir and not args.keep_index:
        import shutil
        if os.path.exists(index_dir):
            shutil.rmtree(index_dir)


def cmd_cluster(args):
    """Perform semantic clustering on data."""
    # Fix FAISS threading issues on some platforms
    import os
    os.environ['OMP_NUM_THREADS'] = '1'

    _, rm, vs = setup_lotus_embeddings(args.embedding_model)
    df = load_data(args.input)

    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Clustering column: {args.column}")
    print(f"Number of clusters: {args.num_clusters}")

    # Create index directory name
    import tempfile
    import os
    index_dir = args.index_dir if args.index_dir else os.path.join(tempfile.gettempdir(), f"lotus_cluster_{os.getpid()}")

    # Index and cluster
    df = df.sem_index(args.column, index_dir)
    result = df.sem_cluster_by(args.column, ncentroids=args.num_clusters)

    # Show cluster distribution
    cluster_counts = result['cluster_id'].value_counts().sort_index()
    print(f"\nCluster distribution:")
    for cluster_id, count in cluster_counts.items():
        print(f"  Cluster {cluster_id}: {count} rows")

    if args.output:
        save_data(result, args.output)
        print(f"\nSaved to {args.output}")
    else:
        print("\nResults:")
        print(result)

    # Clean up temp index if not specified
    if not args.index_dir and not args.keep_index:
        import shutil
        if os.path.exists(index_dir):
            shutil.rmtree(index_dir)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Lotus CLI - Semantic data processing from the command line',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
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
        """
    )

    parser.add_argument('--model', default='gpt-4o-mini', help='LLM model to use (default: gpt-4o-mini)')
    parser.add_argument('--show-usage', action='store_true', help='Show LLM token usage after execution')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Filter command
    filter_parser = subparsers.add_parser('filter', help='Apply semantic filter')
    filter_parser.add_argument('input', help='Input CSV or JSON file')
    filter_parser.add_argument('--condition', '-c', required=True, help='Natural language filter condition')
    filter_parser.add_argument('--output', '-o', help='Output file (prints to stdout if not specified)')

    # Map command
    map_parser = subparsers.add_parser('map', help='Apply semantic map')
    map_parser.add_argument('input', help='Input CSV or JSON file')
    map_parser.add_argument('--instruction', '-i', required=True, help='Natural language mapping instruction')
    map_parser.add_argument('--suffix', '-s', default='_map', help='Suffix for output column (default: _map)')
    map_parser.add_argument('--output', '-o', help='Output file (prints to stdout if not specified)')

    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract structured data')
    extract_parser.add_argument('input', help='Input CSV or JSON file')
    extract_parser.add_argument('--input-cols', required=True, help='Comma-separated input columns')
    extract_parser.add_argument('--fields', '-f', required=True,
                               help='Fields to extract (comma-separated or JSON dict)')
    extract_parser.add_argument('--output', '-o', help='Output file (prints to stdout if not specified)')

    # TopK command
    topk_parser = subparsers.add_parser('topk', help='Rank and select top-k rows')
    topk_parser.add_argument('input', help='Input CSV or JSON file')
    topk_parser.add_argument('--criteria', '-c', required=True, help='Natural language ranking criteria')
    topk_parser.add_argument('--k', '-k', type=int, default=5, help='Number of top results (default: 5)')
    topk_parser.add_argument('--output', '-o', help='Output file (prints to stdout if not specified)')

    # Search command
    search_parser = subparsers.add_parser('search', help='Search web and apply semantic operations')
    search_parser.add_argument('corpus', choices=['arxiv', 'google', 'scholar', 'you', 'bing', 'tavily'],
                               help='Search corpus')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--num', '-n', type=int, default=10, help='Number of results (default: 10)')
    search_parser.add_argument('--rank', '-r', help='Optional ranking criteria for results')
    search_parser.add_argument('--topk', '-k', type=int, help='Number of top results after ranking')
    search_parser.add_argument('--output', '-o', help='Output file (prints to stdout if not specified)')

    # Dedup command
    dedup_parser = subparsers.add_parser('dedup', help='Remove semantic duplicates')
    dedup_parser.add_argument('input', help='Input CSV or JSON file')
    dedup_parser.add_argument('--column', '-c', required=True, help='Column to deduplicate on')
    dedup_parser.add_argument('--threshold', '-t', type=float, default=0.65,
                              help='Similarity threshold for deduplication (default: 0.65)')
    dedup_parser.add_argument('--embedding-model', '-e', default='minishlab/potion-base-8M',
                              help='Embedding model to use (default: minishlab/potion-base-8M)')
    dedup_parser.add_argument('--index-dir', help='Directory to store/load index (uses temp dir if not specified)')
    dedup_parser.add_argument('--keep-index', action='store_true',
                              help='Keep the index after deduplication (only applies to temp indices)')
    dedup_parser.add_argument('--output', '-o', help='Output file (prints to stdout if not specified)')

    # Cluster command
    cluster_parser = subparsers.add_parser('cluster', help='Perform semantic clustering')
    cluster_parser.add_argument('input', help='Input CSV or JSON file')
    cluster_parser.add_argument('--column', '-c', required=True, help='Column to cluster on')
    cluster_parser.add_argument('--num-clusters', '-n', type=int, required=True,
                                help='Number of clusters to create')
    cluster_parser.add_argument('--embedding-model', '-e', default='minishlab/potion-base-8M',
                                help='Embedding model to use (default: minishlab/potion-base-8M)')
    cluster_parser.add_argument('--index-dir', help='Directory to store/load index (uses temp dir if not specified)')
    cluster_parser.add_argument('--keep-index', action='store_true',
                                help='Keep the index after clustering (only applies to temp indices)')
    cluster_parser.add_argument('--output', '-o', help='Output file (prints to stdout if not specified)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == 'filter':
            cmd_filter(args)
        elif args.command == 'map':
            cmd_map(args)
        elif args.command == 'extract':
            cmd_extract(args)
        elif args.command == 'topk':
            cmd_topk(args)
        elif args.command == 'search':
            cmd_search(args)
        elif args.command == 'dedup':
            cmd_dedup(args)
        elif args.command == 'cluster':
            cmd_cluster(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
