import pandas as pd

import lotus
from lotus.models import Model2VecRM
from lotus.vector_store import VicinityVS

# Configure LOTUS with model2vec for fast, lightweight embeddings
rm = Model2VecRM(model="minishlab/potion-base-8M")  # No PyTorch needed!
vs = VicinityVS(backend="BASIC", metric="cosine")  # Pure Python, no external deps!

lotus.settings.configure(rm=rm, vs=vs)

# Sample data with semantic duplicates
data = {
    "Text": [
        "Probability and Random Processes",
        "Optimization Methods in Engineering",
        "Digital Design and Integrated Circuits",
        "Computer Security",
        "I don't know what day it is",
        "I don't know what time it is",  # Semantic duplicate
        "Harry Potter and the Sorcerer's Stone",
        "Machine learning is fascinating",
        "Deep learning is interesting",  # Semantic duplicate
    ]
}
df = pd.DataFrame(data)

print("Original data:")
print(df)
print(f"\nTotal rows: {len(df)}")
print()

# Semantic deduplication using model2vec
# Note: threshold tuned for model2vec embeddings (lower than sentence-transformers)
print("Performing semantic deduplication with Model2VecRM...")
df_dedup = df.sem_index("Text", "dedup_index").sem_dedup("Text", threshold=0.65)

print("\nAfter deduplication:")
print(df_dedup)
print(f"\nRemaining rows: {len(df_dedup)}")
print(f"Removed {len(df) - len(df_dedup)} semantic duplicates")
