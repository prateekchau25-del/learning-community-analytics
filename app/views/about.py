import streamlit as st

import ui


def render(site: str):
    ui.page_header("About this project", "Methods, data and technology")
    st.markdown("""
**Learning Community Analytics using NLP Topics and Knowledge Diffusion Networks** analyses two online learning
communities, **Data Science Stack Exchange** and **Artificial Intelligence Stack Exchange**, to answer three questions:

1. **What do learners discuss, and where do they struggle?** Topic modeling with LDA and BERTopic, validated against the
   communities' own tags, plus a difficulty index (unanswered rate + waiting time + learner confusion).
2. **Who helps whom?** A helper → learner knowledge network with ExpertiseRank, PageRank, betweenness, k-core,
   Louvain communities and rule-based roles, validated against Stack Exchange reputation.
3. **How does knowledge spread?** A time-shuffle social-exposure test, a learner → helper test, Independent Cascade and
   Linear Threshold simulations, CELF influence maximisation, and a check of the simulation against observed reach.

Two tools are built on these results: an **answer predictor** (will a new question be answered?) and an **expert finder**
(who can answer it?). Both are evaluated on future questions (train on the oldest 80%, test on the newest 20%).
""")
    st.markdown("## Pipeline")
    ui.steps([("collect_data.py", "Stack Exchange API, 100 questions with answers and comments per request"),
              ("load_to_postgres.py", "one schema per community + comparison"),
              ("preprocess.py", "cleaning, lemmas, sentiment, confusion"),
              ("topic_model.py", "LDA vs BERTopic, NPMI, NMI"),
              ("network.py", "centrality, communities, roles"),
              ("diffusion.py", "shuffle test, IC/LT, CELF"),
              ("predict.py · recommender.py", "answer model, expert finder"),
              ("compare.py", "Data Science vs AI")])
    st.markdown("## Technology")
    ui.words(["Python", "PostgreSQL", "SQLAlchemy", "pandas", "NLTK", "sentence-transformers", "BERTopic", "UMAP",
              "HDBSCAN", "scikit-learn", "XGBoost", "NetworkX", "SciPy", "Plotly", "PyVis", "Streamlit"])
    st.markdown("## Data")
    st.markdown("Public questions, answers and comments from datascience.stackexchange.com and ai.stackexchange.com, "
                "collected through the Stack Exchange API (content licensed CC BY-SA). People are linked across the two "
                "sites by their public Stack Exchange account id.")
