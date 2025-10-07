#!/usr/bin/env python
"""
Lotus CLI - A command-line interface for semantic data processing

This CLI provides easy access to LOTUS semantic operators for batch data processing.
"""
import json
import sys
from pathlib import Path
from typing import Optional

import typer

# Constants
DEFAULT_LM_MODEL = "gemini/gemini-2.5-flash-lite-preview-09-2025"
DEFAULT_EMBEDDING_MODEL = "minishlab/potion-base-8M"

# Create the main app
app = typer.Typer(
    name="lotus",
    help="Semantic data processing from the command line",
    no_args_is_help=True,
    add_completion=False,
)


def setup_lotus(model: str = DEFAULT_LM_MODEL):
    """Initialize LOTUS with the specified model."""
    import lotus
    from lotus.models import LM

    lm = LM(model=model)
    lotus.settings.configure(lm=lm)
    return lm


def setup_lotus_embeddings(embedding_model: str = DEFAULT_EMBEDDING_MODEL, lm_model: str | None = None):
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


def load_data(input_path: Path):
    """Load data from CSV or JSON file."""
    import pandas as pd

    if not input_path.exists():
        typer.echo(f"Error: Input file not found: {input_path}", err=True)
        raise typer.Exit(1)

    try:
        if input_path.suffix == ".csv":
            return pd.read_csv(input_path)
        elif input_path.suffix == ".json":
            return pd.read_json(input_path)
        else:
            typer.echo(
                f"Error: Unsupported file format: {input_path.suffix}. Use .csv or .json",
                err=True,
            )
            raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error loading file: {e}", err=True)
        raise typer.Exit(1)


def save_data(df, output_path: Path):
    """Save data to CSV or JSON file."""
    try:
        if output_path.suffix == ".csv":
            df.to_csv(output_path, index=False)
        elif output_path.suffix == ".json":
            df.to_json(output_path, orient="records", indent=2)
        else:
            typer.echo(
                f"Error: Unsupported output format: {output_path.suffix}. Use .csv or .json",
                err=True,
            )
            raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error saving file: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def filter(
    input: Path = typer.Argument(..., help="Input CSV or JSON file", exists=True),
    condition: str = typer.Option(..., "--condition", "-c", help="Natural language filter condition"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file (prints to stdout if not specified)"),
    model: str = typer.Option(DEFAULT_LM_MODEL, "--model", help="LLM model to use"),
    show_usage: bool = typer.Option(False, "--show-usage", help="Show LLM token usage after execution"),
):
    """Apply semantic filter to remove rows that don't match a condition.

    \b
    Examples:
      # Filter research papers about AI
      lotus filter papers.csv --condition "{abstract} is about artificial intelligence"

      # Filter and save results
      lotus filter papers.csv -c "{title} is interesting" -o filtered.csv

      # Use a different model
      lotus filter papers.csv -c "{abstract} discusses quantum computing" --model gpt-4o
    """
    try:
        lm = setup_lotus(model)
        df = load_data(input)

        typer.echo(f"Loaded {len(df)} rows from {input}")
        typer.echo(f"Applying filter: {condition}")

        result = df.sem_filter(condition)

        typer.echo(f"Filtered to {len(result)} rows")

        if output:
            save_data(result, output)
            typer.echo(f"Saved to {output}")
        else:
            typer.echo(result.to_string())

        if show_usage:
            lm.print_total_usage()
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def map(
    input: Path = typer.Argument(..., help="Input CSV or JSON file", exists=True),
    instruction: str = typer.Option(..., "--instruction", "-i", help="Natural language mapping instruction"),
    suffix: str = typer.Option("_map", "--suffix", "-s", help="Suffix for output column"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file (prints to stdout if not specified)"),
    model: str = typer.Option(DEFAULT_LM_MODEL, "--model", help="LLM model to use"),
    show_usage: bool = typer.Option(False, "--show-usage", help="Show LLM token usage after execution"),
):
    """Apply semantic map to transform data.

    \b
    Examples:
      # Find similar courses
      lotus map courses.csv --instruction "What is similar to {Course Name}?"

      # Summarize abstracts with custom suffix
      lotus map papers.csv -i "Summarize {abstract}" -s "_summary" -o summaries.csv

      # Extract sentiment
      lotus map reviews.csv -i "What is the sentiment of {review}?" --model gpt-4o
    """
    try:
        lm = setup_lotus(model)
        df = load_data(input)

        typer.echo(f"Loaded {len(df)} rows from {input}")
        typer.echo(f"Applying map: {instruction}")

        result = df.sem_map(instruction, suffix=suffix)

        if output:
            save_data(result, output)
            typer.echo(f"Saved to {output}")
        else:
            typer.echo(result.to_string())

        if show_usage:
            lm.print_total_usage()
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def extract(
    input: Path = typer.Argument(..., help="Input CSV or JSON file", exists=True),
    input_cols: str = typer.Option(..., "--input-cols", help="Comma-separated input columns"),
    fields: str = typer.Option(
        ...,
        "--fields",
        "-f",
        help="Fields to extract (comma-separated or JSON dict)",
    ),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file (prints to stdout if not specified)"),
    model: str = typer.Option(DEFAULT_LM_MODEL, "--model", help="LLM model to use"),
    show_usage: bool = typer.Option(False, "--show-usage", help="Show LLM token usage after execution"),
):
    """Extract structured data from text.

    \b
    Examples:
      # Extract simple fields
      lotus extract people.csv --input-cols description --fields "name,age,occupation"

      # Extract with types (JSON format)
      lotus extract docs.csv --input-cols text --fields '{"name": "string", "count": "int"}'

      # Extract from multiple columns
      lotus extract data.csv --input-cols "title,body" --fields "topic,sentiment,entities" -o extracted.csv
    """
    try:
        lm = setup_lotus(model)
        df = load_data(input)

        typer.echo(f"Loaded {len(df)} rows from {input}")

        # Parse output columns from JSON or simple format
        if fields.startswith("{"):
            output_cols = json.loads(fields)
        else:
            # Simple format: "name,age,location" -> {"name": None, "age": None, "location": None}
            output_cols = {field.strip(): None for field in fields.split(",")}

        typer.echo(f"Extracting fields: {list(output_cols.keys())}")

        result = df.sem_extract(input_cols.split(","), output_cols)

        if output:
            save_data(result, output)
            typer.echo(f"Saved to {output}")
        else:
            typer.echo(result.to_string())

        if show_usage:
            lm.print_total_usage()
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def topk(
    input: Path = typer.Argument(..., help="Input CSV or JSON file", exists=True),
    criteria: str = typer.Option(..., "--criteria", "-c", help="Natural language ranking criteria"),
    k: int = typer.Option(5, "--k", "-k", help="Number of top results"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file (prints to stdout if not specified)"),
    model: str = typer.Option(DEFAULT_LM_MODEL, "--model", help="LLM model to use"),
    show_usage: bool = typer.Option(False, "--show-usage", help="Show LLM token usage after execution"),
):
    """Rank and select top-k rows.

    \b
    Examples:
      # Find top 5 interesting articles
      lotus topk articles.csv --criteria "Which {title} is most interesting?" --k 5

      # Find top 10 relevant papers
      lotus topk papers.csv -c "Which {abstract} is most relevant to quantum computing?" -k 10 -o top.csv

      # Rank courses by difficulty
      lotus topk courses.csv -c "Which {Course Name} is most difficult?" --k 3
    """
    try:
        lm = setup_lotus(model)
        df = load_data(input)

        typer.echo(f"Loaded {len(df)} rows from {input}")
        typer.echo(f"Ranking by: {criteria}")
        typer.echo(f"Selecting top {k} rows")

        result = df.sem_topk(criteria, K=k)

        if output:
            save_data(result, output)
            typer.echo(f"Saved to {output}")
        else:
            typer.echo(result.to_string())

        if show_usage:
            lm.print_total_usage()
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def search(
    corpus: str = typer.Argument(..., help="Search corpus: arxiv, google, scholar, you, bing, tavily"),
    query: str = typer.Argument(..., help="Search query"),
    num: int = typer.Option(10, "--num", "-n", help="Number of results"),
    rank: Optional[str] = typer.Option(None, "--rank", "-r", help="Optional ranking criteria for results"),
    topk: Optional[int] = typer.Option(None, "--topk", "-k", help="Number of top results after ranking"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file (prints to stdout if not specified)"),
    model: str = typer.Option(DEFAULT_LM_MODEL, "--model", help="LLM model to use"),
    show_usage: bool = typer.Option(False, "--show-usage", help="Show LLM token usage after execution"),
):
    """Search web and apply semantic operations.

    \b
    Examples:
      # Search arXiv
      lotus search arxiv "deep learning" --num 10

      # Search and rank results
      lotus search arxiv "machine learning" --num 20 --rank "Which {abstract} is most exciting?" --topk 5

      # Search Google Scholar
      lotus search scholar "quantum computing" -n 15 -o results.csv

      # Search with Tavily
      lotus search tavily "latest AI news" --num 10
    """
    from lotus import WebSearchCorpus, web_search

    try:
        lm = setup_lotus(model)

        # Map corpus string to enum
        corpus_map = {
            "arxiv": WebSearchCorpus.ARXIV,
            "google": WebSearchCorpus.GOOGLE,
            "scholar": WebSearchCorpus.GOOGLE_SCHOLAR,
            "you": WebSearchCorpus.YOU,
            "bing": WebSearchCorpus.BING,
            "tavily": WebSearchCorpus.TAVILY,
        }

        corpus_enum = corpus_map.get(corpus.lower())
        if not corpus_enum:
            typer.echo(
                f"Error: Unknown corpus: {corpus}. Choose from: {list(corpus_map.keys())}",
                err=True,
            )
            raise typer.Exit(1)

        typer.echo(f"Searching {corpus} for: {query}")
        typer.echo(f"Retrieving {num} results")

        df = web_search(corpus_enum, query, num)

        # Apply optional topk ranking
        if rank:
            typer.echo(f"Ranking by: {rank}")
            k = topk if topk else num
            df = df.sem_topk(rank, K=k)

        if output:
            save_data(df, output)
            typer.echo(f"Saved to {output}")
        else:
            typer.echo(df.to_string())

        if show_usage:
            lm.print_total_usage()
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def dedup(
    input: Path = typer.Argument(..., help="Input CSV or JSON file", exists=True),
    column: str = typer.Option(..., "--column", "-c", help="Column to deduplicate on"),
    threshold: float = typer.Option(0.65, "--threshold", "-t", help="Similarity threshold for deduplication"),
    embedding_model: str = typer.Option(DEFAULT_EMBEDDING_MODEL, "--embedding-model", "-e", help="Embedding model to use"),
    index_dir: Optional[Path] = typer.Option(
        None,
        "--index-dir",
        help="Directory to store/load index (uses temp dir if not specified)",
    ),
    keep_index: bool = typer.Option(False, "--keep-index", help="Keep the index after deduplication"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file (prints to stdout if not specified)"),
):
    """Remove semantic duplicates from data.

    \b
    Examples:
      # Deduplicate articles
      lotus dedup articles.csv --column title --threshold 0.65

      # Deduplicate with custom threshold
      lotus dedup papers.csv -c abstract -t 0.8 -o unique.csv

      # Keep index for reuse
      lotus dedup data.csv -c text --index-dir ./my_index --keep-index
    """
    import os
    import shutil
    import tempfile

    try:
        _, rm, vs = setup_lotus_embeddings(embedding_model)
        df = load_data(input)

        typer.echo(f"Loaded {len(df)} rows from {input}")
        typer.echo(f"Deduplicating column: {column}")
        typer.echo(f"Threshold: {threshold}")

        # Create index directory name
        if index_dir:
            index_path = index_dir
        else:
            index_path = Path(tempfile.gettempdir()) / f"lotus_dedup_{os.getpid()}"

        # Index and deduplicate
        df = df.sem_index(column, str(index_path))
        result = df.sem_dedup(column, threshold=threshold)

        removed = len(df) - len(result)
        typer.echo(f"Removed {removed} duplicates ({removed/len(df)*100:.1f}%)")
        typer.echo(f"Remaining: {len(result)} rows")

        if output:
            save_data(result, output)
            typer.echo(f"Saved to {output}")
        else:
            typer.echo(result.to_string())

        # Clean up temp index if not specified
        if not index_dir and not keep_index:
            if index_path.exists():
                shutil.rmtree(index_path)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def cluster(
    input: Path = typer.Argument(..., help="Input CSV or JSON file", exists=True),
    column: str = typer.Option(..., "--column", "-c", help="Column to cluster on"),
    num_clusters: int = typer.Option(..., "--num-clusters", "-n", help="Number of clusters to create"),
    embedding_model: str = typer.Option(DEFAULT_EMBEDDING_MODEL, "--embedding-model", "-e", help="Embedding model to use"),
    index_dir: Optional[Path] = typer.Option(
        None,
        "--index-dir",
        help="Directory to store/load index (uses temp dir if not specified)",
    ),
    keep_index: bool = typer.Option(False, "--keep-index", help="Keep the index after clustering"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file (prints to stdout if not specified)"),
):
    """Perform semantic clustering on data.

    \b
    Examples:
      # Cluster articles into 5 groups
      lotus cluster articles.csv --column text --num-clusters 5

      # Cluster papers with custom model
      lotus cluster papers.csv -c abstract -n 10 -e sentence-transformers/all-MiniLM-L6-v2 -o clustered.csv

      # Cluster and keep index
      lotus cluster data.csv -c description -n 3 --index-dir ./clusters --keep-index
    """
    import os
    import shutil
    import tempfile

    try:
        # Fix FAISS threading issues on some platforms
        os.environ["OMP_NUM_THREADS"] = "1"

        _, rm, vs = setup_lotus_embeddings(embedding_model)
        df = load_data(input)

        typer.echo(f"Loaded {len(df)} rows from {input}")
        typer.echo(f"Clustering column: {column}")
        typer.echo(f"Number of clusters: {num_clusters}")

        # Create index directory name
        if index_dir:
            index_path = index_dir
        else:
            index_path = Path(tempfile.gettempdir()) / f"lotus_cluster_{os.getpid()}"

        # Index and cluster
        df = df.sem_index(column, str(index_path))
        result = df.sem_cluster_by(column, ncentroids=num_clusters)

        # Show cluster distribution
        cluster_counts = result["cluster_id"].value_counts().sort_index()
        typer.echo("\nCluster distribution:")
        for cluster_id, count in cluster_counts.items():
            typer.echo(f"  Cluster {cluster_id}: {count} rows")

        if output:
            save_data(result, output)
            typer.echo(f"\nSaved to {output}")
        else:
            typer.echo("\nResults:")
            typer.echo(result.to_string())

        # Clean up temp index if not specified
        if not index_dir and not keep_index:
            if index_path.exists():
                shutil.rmtree(index_path)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def index(
    input: Path = typer.Argument(..., help="Input CSV or JSON file", exists=True),
    column: str = typer.Option(..., "--column", "-c", help="Column to index"),
    index_dir: Path = typer.Option(..., "--index-dir", "-d", help="Directory to save the index"),
    embedding_model: str = typer.Option(DEFAULT_EMBEDDING_MODEL, "--embedding-model", "-e", help="Embedding model to use"),
):
    """Create a persistent vector index for semantic search.

    \b
    Examples:
      # Create an index for articles
      lotus index articles.csv --column text --index-dir ./article_index

      # Create index with custom embedding model
      lotus index papers.csv -c abstract -d ./paper_index -e sentence-transformers/all-MiniLM-L6-v2
    """
    try:
        _, rm, vs = setup_lotus_embeddings(embedding_model)
        df = load_data(input)

        typer.echo(f"Loaded {len(df)} rows from {input}")
        typer.echo(f"Indexing column: {column}")
        typer.echo(f"Index directory: {index_dir}")

        # Create index
        df = df.sem_index(column, str(index_dir))

        typer.echo(f"✓ Indexed {len(df)} vectors")
        typer.echo(f"✓ Index saved to {index_dir}")
        typer.echo("\nUse 'lotus semsearch' to query this index")
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def semsearch(
    data: Path = typer.Option(..., "--data", "-d", help="Original CSV/JSON file that was indexed", exists=True),
    index_dir: Path = typer.Option(..., "--index-dir", "-i", help="Directory containing the index", exists=True),
    column: str = typer.Option(..., "--column", "-c", help="Column that was indexed"),
    query: str = typer.Option(..., "--query", "-q", help="Search query"),
    k: int = typer.Option(5, "--k", "-k", help="Number of results to return"),
    embedding_model: str = typer.Option(DEFAULT_EMBEDDING_MODEL, "--embedding-model", "-e", help="Embedding model to use"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file (prints to stdout if not specified)"),
):
    """Search a pre-built index with semantic queries.

    \b
    Examples:
      # Search an index
      lotus semsearch --data articles.csv --index-dir ./article_index --column text --query "machine learning" --k 5

      # Search and save results
      lotus semsearch -d papers.csv -i ./paper_index -c abstract -q "quantum computing" -k 10 -o results.csv
    """
    try:
        _, rm, vs = setup_lotus_embeddings(embedding_model)
        df = load_data(data)

        typer.echo(f"Loaded {len(df)} rows from {data}")
        typer.echo(f"Loading index from: {index_dir}")
        typer.echo(f"Query: {query}")
        typer.echo(f"Top K: {k}")

        # Load index and search
        df = df.sem_index(column, str(index_dir))
        result = df.sem_search(column, query, K=k)

        typer.echo(f"\nFound {len(result)} results")

        if output:
            save_data(result, output)
            typer.echo(f"Saved to {output}")
        else:
            typer.echo("\nResults:")
            typer.echo(result.to_string())
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="sim-join")
def sim_join(
    left: Path = typer.Option(..., "--left", "-l", help="Left dataset (CSV or JSON file)", exists=True),
    right: Path = typer.Option(..., "--right", "-r", help="Right dataset (CSV or JSON file)", exists=True),
    left_col: str = typer.Option(..., "--left-col", help="Column name in left dataset"),
    right_col: str = typer.Option(..., "--right-col", help="Column name in right dataset"),
    k: int = typer.Option(1, "--k", "-k", help="Number of matches per left row"),
    embedding_model: str = typer.Option(DEFAULT_EMBEDDING_MODEL, "--embedding-model", "-e", help="Embedding model to use"),
    left_index: Optional[Path] = typer.Option(
        None,
        "--left-index",
        help="Directory for left dataset index (temp if not specified)",
    ),
    right_index: Optional[Path] = typer.Option(
        None,
        "--right-index",
        help="Directory for right dataset index (temp if not specified)",
    ),
    keep_index: bool = typer.Option(False, "--keep-index", help="Keep indices after join"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file (prints to stdout if not specified)"),
):
    """Perform fuzzy similarity join between two datasets.

    \b
    Examples:
      # Join products by description
      lotus sim-join --left products.csv --right reviews.csv --left-col description --right-col review_text

      # Join with multiple matches
      lotus sim-join -l articles.csv -r references.csv --left-col title --right-col ref_title -k 3 -o joined.csv

      # Keep indices for reuse
      lotus sim-join -l data1.csv -r data2.csv --left-col text --right-col content --left-index ./idx1 --right-index ./idx2 --keep-index
    """
    import os
    import shutil
    import tempfile

    try:
        _, rm, vs = setup_lotus_embeddings(embedding_model)

        # Load both datasets
        left_df = load_data(left)
        right_df = load_data(right)

        typer.echo(f"Loaded left dataset: {len(left_df)} rows from {left}")
        typer.echo(f"Loaded right dataset: {len(right_df)} rows from {right}")
        typer.echo(f"Joining on: {left_col} ~ {right_col}")
        typer.echo(f"Top K matches per row: {k}")

        # Create index directories
        if left_index:
            left_index_path = left_index
        else:
            left_index_path = Path(tempfile.gettempdir()) / f"lotus_left_{os.getpid()}"

        if right_index:
            right_index_path = right_index
        else:
            right_index_path = Path(tempfile.gettempdir()) / f"lotus_right_{os.getpid()}"

        # Index both datasets
        typer.echo("\nIndexing datasets...")
        left_df = left_df.sem_index(left_col, str(left_index_path))
        right_df = right_df.sem_index(right_col, str(right_index_path))

        # Perform similarity join
        typer.echo("Performing similarity join...")
        result = left_df.sem_sim_join(right_df, left_col, right_col, K=k)

        typer.echo(f"\nJoin complete: {len(result)} result rows")

        if output:
            save_data(result, output)
            typer.echo(f"Saved to {output}")
        else:
            typer.echo("\nResults (first 10 rows):")
            typer.echo(result.head(10).to_string())

        # Clean up temp indices
        if not left_index and not keep_index:
            if left_index_path.exists():
                shutil.rmtree(left_index_path)
        if not right_index and not keep_index:
            if right_index_path.exists():
                shutil.rmtree(right_index_path)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="list")
def list_commands():
    """List all available commands.

    \b
    This shows all semantic operations available in Lotus CLI.
    Use 'lotus COMMAND --help' for detailed information about each command.
    """
    commands = [
        ("filter", "Apply semantic filter to data"),
        ("map", "Apply semantic map to transform data"),
        ("extract", "Extract structured data from text"),
        ("topk", "Rank and select top-k rows"),
        ("search", "Search web and apply semantic operations"),
        ("dedup", "Remove semantic duplicates from data"),
        ("cluster", "Perform semantic clustering on data"),
        ("index", "Create a persistent vector index"),
        ("semsearch", "Search a pre-built index"),
        ("sim-join", "Perform fuzzy similarity join between datasets"),
    ]

    typer.echo("Available Commands:")
    typer.echo("")
    for cmd, desc in commands:
        typer.echo(f"  {cmd:<12} {desc}")
    typer.echo("")
    typer.echo("Use 'lotus COMMAND --help' for more information about a command.")


def main():
    """Main entry point for the CLI."""
    try:
        app()
    except KeyboardInterrupt:
        typer.echo("\nOperation cancelled by user.", err=True)
        sys.exit(130)
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
