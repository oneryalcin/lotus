import pandas as pd

import lotus
from lotus.models import LM, CrossEncoderReranker, Model2VecRM
from lotus.vector_store import FaissVS

# Configure LOTUS with model2vec for fast, lightweight embeddings
lm = LM(model="gpt-4o-mini")
rm = Model2VecRM(model="minishlab/potion-base-8M")  # Static embeddings, no PyTorch!
reranker = CrossEncoderReranker(model="mixedbread-ai/mxbai-rerank-large-v1")

lotus.settings.configure(lm=lm, rm=rm, reranker=reranker, vs=FaissVS())

# Sample data
data = {
    "Course Name": [
        "Probability and Random Processes",
        "Optimization Methods in Engineering",
        "Digital Design and Integrated Circuits",
        "Computer Security",
        "Introduction to Computer Science",
        "Introduction to Data Science",
        "Introduction to Machine Learning",
        "Introduction to Artificial Intelligence",
        "Introduction to Robotics",
        "Introduction to Computer Vision",
        "Introduction to Natural Language Processing",
        "Introduction to Reinforcement Learning",
        "Introduction to Deep Learning",
        "Introduction to Computer Networks",
    ]
}
df = pd.DataFrame(data)

print("Using Model2VecRM for semantic search (no PyTorch, fast startup!)")
print(f"Model: {rm.model}")
print()

# Index and search
df = df.sem_index("Course Name", "index_dir_model2vec").sem_search(
    "Course Name",
    "Which course name is most related to computer security?",
    K=8,
    n_rerank=4,
)

print("Search results:")
print(df)
print()

# Compare to filtering (LLM-based)
data2 = pd.DataFrame({"Course Name": [
    "Computer Security",
    "Cybersecurity Fundamentals",
    "Network Security",
    "Information Security",
]})

filtered = data2.sem_filter("{Course Name} is about protecting computer systems")
print("\nFiltered results (LLM-based):")
print(filtered)
