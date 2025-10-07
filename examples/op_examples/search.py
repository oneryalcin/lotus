import pandas as pd

import lotus
from lotus.models import LM, CrossEncoderReranker, Model2VecRM
from lotus.vector_store import FaissVS

lm = LM(model="gpt-4o-mini")
rm = Model2VecRM(model="minishlab/potion-base-8M")
reranker = CrossEncoderReranker(model="mixedbread-ai/mxbai-rerank-large-v1")
vs = FaissVS()

lotus.settings.configure(lm=lm, rm=rm, reranker=reranker, vs=vs)
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

df = df.sem_index("Course Name", "index_dir").sem_search(
    "Course Name",
    "Which course name is most related to computer security?",
    K=8,
    n_rerank=4,
)
print(df)
