# Learning Community Analytics using NLP Topics and Knowledge Diffusion Networks

Capstone project that analyses an online learning community, the
[CS50 Stack Exchange](https://cs50.stackexchange.com) Q&A site (4,000 questions,
4,253 answers, 7,168 comments, 3,003 users, 2019–2026), to answer three questions:

1. **What do learners discuss, and what do they struggle with?** NLP topic modeling (LDA vs BERTopic)
2. **Who helps whom?** A knowledge network with centrality, communities and user roles
3. **How does knowledge spread?** Diffusion tests on real data plus IC/LT simulation and influence maximisation

It also includes two applications built on these results: an **answer predictor**
(will a new question get answered?) and an **expert finder** (who can answer it?),
all shown in an interactive **Streamlit dashboard** backed by **PostgreSQL**.

## Headline results

| Area | Result |
|---|---|
| Topics | BERTopic (19 topics) beats LDA (20 topics): NPMI coherence 0.31 vs 0.24, diversity 0.87 vs 0.73, agreement with course tags (NMI) 0.56 vs 0.38 |
| Hardest topic | *Tideman* (Week 3), the problem set CS50 students widely call the hardest |
| Network | 2,891 users; the top 1% of users give 81% of all help; 16 communities of 10+ users (modularity 0.60) |
| Validation | ExpertiseRank's top 20 overlaps 85% with the top 20 by accepted answers |
| Diffusion | Modest social-influence effect overall (Stouffer z = 2.68, p = 0.004), significant for Finance, DNA and SQL |
| Simulation | Simulated IC influence matches observed time-respecting reach (Spearman ρ = 0.86) |
| Prediction | Random Forest, ROC-AUC 0.675 on future questions (baseline 0.50) |
| Expert finder | A real answerer is in the top 10 for 81% of future questions (popularity baseline 70%) |

## Workflow

```
Stack Exchange API ─► data/raw/*.csv ─► PostgreSQL (pgAdmin)
                                  │
                                  ▼
   preprocess ─► topic_model ─► network ─► diffusion ─► predict ─► recommender ─► figures
   (text clean)  (LDA/BERTopic)  (who helps)  (spread)   (answered?)  (expert finder)  (PNG)
                                  │
                                  ▼
                outputs/ ─► PostgreSQL result tables ─► Streamlit dashboard
```

| Step | Script | What it does |
|---|---|---|
| 1 | `src/collect_data.py` | Downloads questions, answers and comments from the Stack Exchange API |
| 2 | `src/load_to_postgres.py` | Creates the tables in PostgreSQL and loads the raw data **and** the results |
| 3 | `src/preprocess.py` | Cleans HTML/code, lemmatises text, adds sentiment, confusion and error features |
| 4 | `src/topic_model.py` | LDA (k chosen by coherence) vs BERTopic, topic difficulty, topic trends |
| 5 | `src/network.py` | Helper → learner network: ExpertiseRank, PageRank, betweenness, k-core, Louvain communities, roles |
| 6 | `src/diffusion.py` | Exposure (shuffle) test, learner → helper test, IC/LT simulation, CELF influence maximisation, validation |
| 7 | `src/predict.py` | Answer prediction (Logistic Regression, Random Forest, XGBoost), temporal split |
| 8 | `src/recommender.py` | Expert finder (content + authority + recency), Hit@k / MRR evaluation |
| 9 | `src/figures.py` | 16 report figures in `outputs/figures/` |
| – | `src/run_pipeline.py` | Runs steps 3–9 in order |
| – | `app/streamlit_app.py` | Interactive dashboard (7 pages) |

The repository already contains the downloaded data (`data/`) and all results
(`outputs/`), so the dashboard works right after cloning. Re-running the pipeline is optional.

## Setup on the laptop (one time)

### 1. Install the software
- **Python 3.12** from https://www.python.org/downloads/.
  During setup, tick **"Add Python to PATH"**.
- **PostgreSQL** from https://www.postgresql.org/download/windows/.
  The installer includes **pgAdmin**. Write down the password you choose for the `postgres` user.
- **Git** from https://git-scm.com/download/win.

### 2. Create the database
Open **pgAdmin**, expand *Servers → PostgreSQL*, enter your password, then
right-click **Databases → Create → Database…**, name it `learning_community` and click **Save**.

### 3. Get the project and install the libraries
Open **Command Prompt** and run:
```
cd %USERPROFILE%\Documents
git clone https://github.com/prateekchau25-del/learning-community-analytics.git
cd learning-community-analytics
pip install -r requirements.txt
```
The install downloads about 1 GB (PyTorch for the language model) and takes 5–15 minutes.

### 4. Connect the database
```
copy .env.example .env
notepad .env
```
In Notepad, replace `your_postgres_password_here` with your PostgreSQL password and save.

### 5. Load everything into PostgreSQL
```
python src/load_to_postgres.py
```
This creates 16 tables: 5 raw tables (users, questions, answers, comments, interactions)
and 11 result tables (topics, question_topics, user_metrics, help_edges, …).

### 6. Start the dashboard
```
streamlit run app/streamlit_app.py
```
Your browser opens at http://localhost:8501. The sidebar should say
**Data source: PostgreSQL**. The first visit to *Answer prediction* or *Expert finder*
downloads a 90 MB language model once.

## Showing the database in pgAdmin
- Refresh *learning_community → Schemas → public → Tables* to see all tables.
- Open **Tools → Query Tool**, then **File → Open** `sql/queries.sql`, select one query and press **F5**.
  Queries 1–6 use the raw data, 7–11 the analysis results.
- The dashboard's **Database (SQL)** page runs the same queries.

## Re-running the analysis (optional)
```
python src/run_pipeline.py            # everything, about 10 minutes
python src/run_pipeline.py --from network
python src/load_to_postgres.py        # reload the new results into PostgreSQL
```
With `.env` configured, the pipeline reads its input from PostgreSQL; otherwise from `data/raw/`.

To download fresh data (the free API allows 300 requests per day, enough for about 4,000 questions):
```
python src/collect_data.py --site cs50 --max-questions 4000
```

## Getting updates later
```
cd %USERPROFILE%\Documents\learning-community-analytics
git pull
pip install -r requirements.txt
python src/load_to_postgres.py
```

## Project structure
```
app/streamlit_app.py        dashboard
src/                        pipeline (see table above) + config.py, db.py, data_loader.py
sql/schema.sql              raw table definitions (keys, indexes)
sql/queries.sql             11 analysis queries for pgAdmin
data/raw/cs50/              downloaded CSVs
data/processed/             cleaned questions, embeddings, topic assignments, features
outputs/                    result tables (CSV/JSON)
outputs/figures/            report figures (PNG)
docs/                       project report and presentation
```

## Troubleshooting
| Error | Fix |
|---|---|
| `'python' is not recognized` | Reinstall Python with "Add Python to PATH" ticked, then open a new Command Prompt. |
| `password authentication failed` | The password in `.env` doesn't match your PostgreSQL password. |
| `database "learning_community" does not exist` | Do step 2 again. |
| `connection refused` | PostgreSQL isn't running: open *Services*, start **postgresql-x64-…**. |
| Sidebar says *Data source: CSV files* | `.env` is missing or wrong; the dashboard still works from the CSV files. Fix `.env` and restart. |
| `streamlit` is not recognized | Use `python -m streamlit run app/streamlit_app.py`. |
| `API error 502: throttle violation` | Daily API limit reached. Wait until tomorrow, or use fewer `--max-questions`. |
