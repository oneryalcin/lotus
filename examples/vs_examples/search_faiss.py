import pandas as pd

import lotus
from lotus.models import Model2VecRM
from lotus.vector_store import FaissVS

rm = Model2VecRM(model="minishlab/potion-base-8M")
vs = FaissVS()

lotus.settings.configure(rm=rm, vs=vs)
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
    "Which course name is most related to machine learning?",
    K=8,
)
print(df)
