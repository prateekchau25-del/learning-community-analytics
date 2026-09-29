# Learning Community Analytics using NLP Topics and Knowledge Diffusion Networks

Capstone project that analyses two online learning communities, **Data Science Stack Exchange**
and **Artificial Intelligence Stack Exchange**, to answer three questions:

1. **What do learners discuss, and where do they struggle?** NLP topic modeling (LDA vs BERTopic),
   validated against the communities' own tags, plus a topic difficulty index.
2. **Who helps whom?** A knowledge network with experts, bridges, communities and roles.
3. **How does knowledge spread?** Diffusion tests on real data, Independent Cascade / Linear Threshold
   simulation and influence maximisation, checked against what really happened.

On top of the analysis sit two tools, an **answer predictor** (will a new question get answered?) and
an **expert finder** (who can answer it?), plus a **Data Science vs AI comparison**, including the people
who are active in both communities. Everything is shown in an interactive **Streamlit web app** backed by
**PostgreSQL**.

## Key results

| | Data Science SE | AI SE |
|---|---|---|
| Topics found (BERTopic) | 19 | 19 |
| BERTopic vs LDA, agreement with tags (NMI) | 0.27 vs 0.19 | 0.39 vs 0.27 |
| Share of help given by the top 1% of helpers | 67% | 61% |
| Questions answered / median time to answer | 60.2% / 7.2 h | 70.6% / 9.5 h |
| Contact spreads topics (exposure test, topics significant) | 0 / 19 (z = −2.60) | 0 / 19 (z = −5.87) |
| Answered learners who later become helpers | 2.8% vs 2.9% if unanswered (n.s.) | 7.6% vs 5.4% if unanswered (p = 0.002) |
| Simulated vs observed reach (Spearman ρ) | 0.55 | 0.03 |
| Will a question be answered? Best ROC-AUC | 0.637 (Logistic Regression) | 0.649 (XGBoost) |
| Expert finder Hit@10: hybrid vs popularity | 35.8% vs 19.1% | 49.8% vs 15.4% |

Across the two sites, 1,011 people have accounts in both, and 247 of them answer in both. These 247 people give
44.7% of the help in Data Science and 32.4% of the help in AI.

## Workflow

```
Stack Exchange API ─► data/raw/<site>/ ─► PostgreSQL (schema per community)
                                  │
                                  ▼
 preprocess ─► topic_model ─► network ─► diffusion ─► predict ─► recommender ─► figures
                                  │                  (run for datascience and for ai)
                                  ▼
                              compare  ─►  outputs/comparison/
                                  │
                                  ▼
                  PostgreSQL result tables ─► Streamlit web app
```

| Step | Script | What it does |
|---|---|---|
| 1 | `src/collect_data.py` | Downloads questions with their answers and comments (100 per API request; resumes if interrupted) |
| 2 | `src/load_to_postgres.py` | Creates one schema per community (`datascience`, `ai`) plus `comparison`, and loads data and results |
| 3 | `src/preprocess.py` | Cleans HTML and code, lemmatises text, adds sentiment, confusion, error and maths features |
| 4 | `src/topic_model.py` | LDA vs BERTopic, validated against 10 tag categories (GenAI, NLP, Vision, RL, …) |
| 5 | `src/network.py` | Helper → learner network: ExpertiseRank, PageRank, betweenness, k-core, Louvain, roles |
| 6 | `src/diffusion.py` | Exposure (shuffle) test, learner → helper test, IC/LT simulation, CELF, validation |
| 7 | `src/predict.py` | Answer prediction (Logistic Regression, Random Forest, XGBoost), temporal split |
| 8 | `src/recommender.py` | Expert finder (content + authority + recency), Hit@k / MRR |
| 9 | `src/figures.py` | Report figures per community in `outputs/<site>/figures/` |
| 10 | `src/compare.py` | Data Science vs AI: measures, statistical tests, topic mix, shared people |
| – | `src/run_pipeline.py` | Runs steps 3–10 for both communities |
| – | `app/streamlit_app.py` | Web app: Home, Topics, Network, Diffusion, DS vs AI, Question assistant, Database, About |

The repository contains the downloaded data and all results, so the web app works right after cloning.

## Setup on the laptop

### 1. Install the software (skip what you already have)
- **Python 3.12** (or newer) from https://www.python.org/downloads/, with **"Add Python to PATH"** ticked.
- **PostgreSQL** with **pgAdmin** from https://www.postgresql.org/download/windows/.
- **Git** from https://git-scm.com/download/win.

### 2. Create the database
In **pgAdmin**: right-click **Databases → Create → Database…**, name it `learning_community`, click **Save**.

### 3. Get the project and install the libraries
```
git clone https://github.com/prateekchau25-del/learning-community-analytics.git
cd learning-community-analytics
python -m pip install -r requirements.txt
```
(Already have the folder? Run `git pull` inside it instead of `git clone`.)

### 4. Connect the database
```
copy .env.example .env
notepad .env
```
Set `DB_PASSWORD=` to your PostgreSQL password and save.

### 5. Load everything into PostgreSQL
```
python src/load_to_postgres.py
```
This creates three schemas: `datascience` and `ai` (16 tables each: raw data + results) and `comparison`.

### 6. Start the web app
```
streamlit run app/streamlit_app.py
```
The browser opens at http://localhost:8501. The header shows **Data source: PostgreSQL**, and the switch
at the top right changes the community.

## pgAdmin
- Tables: *learning_community → Schemas → datascience / ai / comparison → Tables*.
- Queries: **Tools → Query Tool**, open `sql/queries.sql`. For queries 1–10 first run
  `SET search_path TO datascience;` (or `ai`). Queries 11–14 compare both communities.

## Re-running the analysis (optional)
```
python src/run_pipeline.py                         # both communities + comparison (~25 minutes)
python src/run_pipeline.py --site ai --from network
python src/load_to_postgres.py                     # reload the new results
```
Fresh data (the API allows 300 requests per day without a key; a free key from
https://stackapps.com/apps/oauth/register gives 10,000; put it in `.env` as `SE_API_KEY=`):
```
python src/collect_data.py --site datascience --max-questions 10000
python src/collect_data.py --site ai --max-questions 10000
```

## Project structure
```
app/streamlit_app.py, app/ui.py, app/data.py, app/views/   web app
src/                        pipeline (table above) + config.py, db.py, data_loader.py
sql/schema.sql              table definitions (keys, indexes)
sql/queries.sql             14 analysis queries for pgAdmin
data/raw/<site>/            downloaded data (CSV)
data/processed/<site>/      cleaned questions, embeddings, topics, features
outputs/<site>/             results and figures per community
outputs/comparison/         Data Science vs AI
```

## Troubleshooting
| Problem | Fix |
|---|---|
| `'python' is not recognized` | Reinstall Python with "Add Python to PATH" ticked, open a new Command Prompt. |
| `No module named ...` | `python -m pip install -r requirements.txt` |
| `password authentication failed` | The password in `.env` does not match PostgreSQL. |
| `database "learning_community" does not exist` | Do step 2 again. |
| `connection refused` | Start the **postgresql-x64-…** service in *Services*. |
| Header says *Data source: CSV files* | `.env` missing or wrong; the app still works from the CSV files. |
| `streamlit` is not recognized | `python -m streamlit run app/streamlit_app.py` |
