#!/usr/bin/env python
"""
Lotus Pipeline Runner - Execute multi-step semantic data processing pipelines

This tool allows you to define complex data processing workflows in YAML and execute them.
"""
import argparse
import sys
from pathlib import Path
from typing import Any, Dict


class PipelineRunner:
    """Execute a LOTUS pipeline defined in YAML."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.variables: Dict[str, Any] = {}
        self.lm = None
        self.rm = None
        self.vs = None
        self.index_dirs: Dict[str, str] = {}  # Track index directories

    def setup(self):
        """Initialize LOTUS with the specified configuration."""
        import lotus
        from lotus.models import LM

        model_config = self.config.get('model', {})
        model_name = model_config.get('name', 'gpt-4o-mini')
        embedding_model = model_config.get('embedding_model')

        self.lm = LM(model=model_name)

        # Setup embeddings if specified
        if embedding_model:
            from lotus.models import Model2VecRM
            from lotus.vector_store import VicinityVS

            self.rm = Model2VecRM(model=embedding_model)
            self.vs = VicinityVS(backend="BASIC", metric="cosine")
            lotus.settings.configure(lm=self.lm, rm=self.rm, vs=self.vs)
        else:
            lotus.settings.configure(lm=self.lm)

    def load_data(self, step: Dict[str, Any]):
        """Load data from a file."""
        import pandas as pd

        source = step['source']
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")

        if path.suffix == '.csv':
            return pd.read_csv(source)
        elif path.suffix == '.json':
            return pd.read_json(source)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def save_data(self, df, destination: str):
        """Save data to a file."""
        import pandas as pd

        path = Path(destination)

        if path.suffix == '.csv':
            df.to_csv(destination, index=False)
        elif path.suffix == '.json':
            df.to_json(destination, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def execute_step(self, step: Dict[str, Any]):
        """Execute a single pipeline step."""
        step_type = step['type']

        print(f"Executing {step_type}: {step.get('name', 'unnamed')}")

        # Get input dataframe
        if 'input' in step:
            df = self.variables[step['input']]
        elif 'source' in step:
            df = self.load_data(step)
        else:
            raise ValueError("Step must have either 'input' or 'source'")

        # Execute the appropriate operation
        if step_type == 'filter':
            result = df.sem_filter(step['condition'])

        elif step_type == 'map':
            suffix = step.get('suffix', '_map')
            result = df.sem_map(step['instruction'], suffix=suffix)

        elif step_type == 'extract':
            input_cols = step['input_cols']
            if isinstance(input_cols, str):
                input_cols = [input_cols]
            result = df.sem_extract(input_cols, step['output_cols'])

        elif step_type == 'topk':
            k = step.get('k', 5)
            result = df.sem_topk(step['criteria'], K=k)

        elif step_type == 'join':
            right_df = self.variables[step['right']]
            result = df.sem_join(right_df, step['condition'])

        elif step_type == 'search':
            from lotus import WebSearchCorpus, web_search

            corpus_map = {
                'arxiv': WebSearchCorpus.ARXIV,
                'google': WebSearchCorpus.GOOGLE,
                'scholar': WebSearchCorpus.GOOGLE_SCHOLAR,
                'you': WebSearchCorpus.YOU,
                'bing': WebSearchCorpus.BING,
                'tavily': WebSearchCorpus.TAVILY,
            }

            corpus = corpus_map[step['corpus'].lower()]
            num_results = step.get('num_results', 10)
            result = web_search(corpus, step['query'], num_results)

        elif step_type == 'aggregate':
            result = pd.DataFrame({'result': [df.sem_agg(step['instruction'])._output[0]]})

        elif step_type == 'index':
            if not self.rm or not self.vs:
                raise ValueError("Embedding model not configured. Add 'embedding_model' to pipeline config.")
            index_dir = step['index_dir']
            column = step['column']
            result = df.sem_index(column, index_dir)
            # Store index dir both by step name and column name for lookup
            if 'name' in step:
                self.index_dirs[step['name']] = index_dir
            self.index_dirs[column] = index_dir

        elif step_type == 'dedup':
            if not self.rm or not self.vs:
                raise ValueError("Embedding model not configured. Add 'embedding_model' to pipeline config.")
            column = step['column']
            threshold = step.get('threshold', 0.65)
            index_dir = step.get('index_dir', self.index_dirs.get(column))
            if not index_dir:
                raise ValueError(f"No index found for column '{column}'. Add 'index' step first or specify 'index_dir'.")
            df = df.sem_index(column, index_dir)
            result = df.sem_dedup(column, threshold=threshold)

        elif step_type == 'cluster':
            if not self.rm or not self.vs:
                raise ValueError("Embedding model not configured. Add 'embedding_model' to pipeline config.")
            # Fix FAISS threading issues
            import os
            os.environ['OMP_NUM_THREADS'] = '1'

            column = step['column']
            num_clusters = step['num_clusters']

            # Reset index to ensure continuous 0-based indexing (required for clustering)
            df = df.reset_index(drop=True)

            # Create a new index for this dataframe (can't reuse old index with different rows)
            import tempfile
            cluster_index = tempfile.mkdtemp(prefix='pipeline_cluster_')
            df = df.sem_index(column, cluster_index)
            result = df.sem_cluster_by(column, ncentroids=num_clusters)

        elif step_type == 'semsearch':
            if not self.rm or not self.vs:
                raise ValueError("Embedding model not configured. Add 'embedding_model' to pipeline config.")
            column = step['column']
            query = step['query']
            k = step.get('k', 5)
            index_dir = step.get('index_dir', self.index_dirs.get(column))
            if not index_dir:
                raise ValueError(f"No index found for column '{column}'. Add 'index' step first or specify 'index_dir'.")
            df = df.sem_index(column, index_dir)
            result = df.sem_search(column, query, K=k)

        elif step_type == 'sim_join':
            if not self.rm or not self.vs:
                raise ValueError("Embedding model not configured. Add 'embedding_model' to pipeline config.")
            right_df = self.variables[step['right']]
            left_col = step['left_col']
            right_col = step['right_col']
            k = step.get('k', 1)

            # Index both if needed
            left_index = step.get('left_index', self.index_dirs.get(left_col))
            right_index = step.get('right_index', self.index_dirs.get(right_col))

            if not left_index:
                import tempfile
                left_index = tempfile.mkdtemp(prefix='pipeline_left_')
            if not right_index:
                import tempfile
                right_index = tempfile.mkdtemp(prefix='pipeline_right_')

            df = df.sem_index(left_col, left_index)
            right_df = right_df.sem_index(right_col, right_index)
            result = df.sem_sim_join(right_df, left_col, right_col, K=k)

        else:
            raise ValueError(f"Unknown step type: {step_type}")

        print(f"  Result: {len(result)} rows")
        return result

    def run(self):
        """Execute the entire pipeline."""
        self.setup()

        steps = self.config.get('steps', [])
        if not steps:
            raise ValueError("Pipeline must have at least one step")

        for step in steps:
            result = self.execute_step(step)

            # Store result if the step has a name
            if 'name' in step:
                self.variables[step['name']] = result

            # Save output if specified
            if 'output' in step:
                self.save_data(result, step['output'])
                print(f"  Saved to: {step['output']}")

        if self.config.get('show_usage', False):
            self.lm.print_total_usage()


def main():
    """Main entry point for pipeline runner."""
    parser = argparse.ArgumentParser(
        description='Lotus Pipeline Runner - Execute YAML-defined semantic data processing pipelines',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Pipeline YAML:

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
        """
    )

    parser.add_argument('pipeline', help='Path to pipeline YAML file')
    parser.add_argument('--dry-run', action='store_true', help='Parse and validate pipeline without executing')
    parser.add_argument('--show-usage', action='store_true', help='Show LLM token usage after execution')

    args = parser.parse_args()

    try:
        import yaml

        # Load pipeline configuration
        with open(args.pipeline, 'r') as f:
            config = yaml.safe_load(f)

        if args.show_usage:
            config['show_usage'] = True

        if args.dry_run:
            print("Pipeline configuration:")
            print(yaml.dump(config, indent=2))
            print("\nPipeline is valid!")
            return

        # Execute pipeline
        runner = PipelineRunner(config)
        runner.run()

        print("\nPipeline completed successfully!")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
