# Learning Community Analytics using NLP Topics and Knowledge Diffusion Networks

Capstone project that analyses an online learning community, the
[CS50 Stack Exchange](https://cs50.stackexchange.com) Q&A site, to find
**what learners discuss** (NLP topic modeling) and **how knowledge spreads
between them** (network analysis and diffusion modeling).

## Project workflow

```
Stack Exchange API -> CSV files -> PostgreSQL -> NLP topics + knowledge network + diffusion -> Dashboard
```

| Step | Script | Status |
|---|---|---|
| 1. Data collection | `src/collect_data.py` | Done |
| 2. Load into PostgreSQL | `src/load_to_postgres.py` | Done |
| 3. SQL analysis (pgAdmin) | `sql/queries.sql` | Done |
| 4. Text preprocessing | | To do |
| 5. Topic modeling (LDA / BERTopic) | | To do |
| 6. Knowledge network + centrality | | To do |
| 7. Diffusion modeling | | To do |
| 8. Streamlit dashboard | | To do |

## Project structure

```
src/collect_data.py      Download questions, answers, comments from Stack Exchange
src/db.py                PostgreSQL connection (reads .env)
src/load_to_postgres.py  Create tables and load the CSVs
sql/schema.sql           Table definitions (keys, indexes)
sql/queries.sql          Analysis queries for pgAdmin
data/raw/<site>/         Downloaded CSVs (not stored in git)
```

## Setup on a new laptop (one time)

### 1. Install the software
- **Python 3.11 or 3.12** from https://www.python.org/downloads/.
  During setup, tick **"Add Python to PATH"**.
- **PostgreSQL** from https://www.postgresql.org/download/windows/.
  The installer includes **pgAdmin**. Write down the password you choose for the `postgres` user.
- **Git** from https://git-scm.com/download/win.

### 2. Create the database
Open **pgAdmin**, expand *Servers -> PostgreSQL*, enter your password, then
right-click **Databases -> Create -> Database...**, name it `learning_community` and click Save.

### 3. Get the project
Open **Command Prompt** and run:
```
cd %USERPROFILE%\Documents
git clone https://github.com/prateekchau25-del/learning-community-analytics.git
cd learning-community-analytics
pip install -r requirements.txt
copy .env.example .env
notepad .env
```
In Notepad, replace `your_postgres_password_here` with your PostgreSQL password and save.

### 4. Download the data and load it into PostgreSQL
```
python src/collect_data.py --site cs50 --max-questions 3000
python src/load_to_postgres.py --site cs50
```
The download takes a few minutes. The free API limit is 300 requests per day,
which is enough for this command about twice a day.

### 5. Look at the data in pgAdmin
- Refresh *learning_community -> Schemas -> public -> Tables* to see the 5 tables.
- Open **Tools -> Query Tool**, open `sql/queries.sql`, select one query and press **F5**.

## Getting updates later
```
cd %USERPROFILE%\Documents\learning-community-analytics
git pull
pip install -r requirements.txt
```

## Troubleshooting
| Error | Fix |
|---|---|
| `'python' is not recognized` | Reinstall Python with "Add Python to PATH" ticked, then open a new Command Prompt. |
| `password authentication failed` | The password in `.env` doesn't match your PostgreSQL password. |
| `database "learning_community" does not exist` | Do step 2 again. |
| `connection refused` | PostgreSQL isn't running: open *Services*, start **postgresql-x64-…**. |
| `API error 502: throttle violation` | Daily API limit reached. Wait until tomorrow, or use fewer `--max-questions`. |
