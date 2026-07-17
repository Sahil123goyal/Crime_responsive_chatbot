# Text-to-SQL Chatbot (MySQL + LangChain + Groq)

Chat with your MySQL database in plain English. Ask a question, the LLM writes
the SQL, runs it against your DB, and gives you a natural-language answer +
the raw SQL/result for transparency.

## 1. Install dependencies

```bash
cd text2sql-chatbot
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

## 2. Get a free Groq API key

Sign up at https://console.groq.com and grab an API key (free tier is generous
and fast — good for this use case).

## 3. Configure credentials

Either:
- Copy `.env.example` to `.env` and fill in your MySQL + Groq details, **or**
- Just type them into the sidebar when the app runs (simplest for a quick test).

## 4. Run it

```bash
streamlit run app.py
```

Open the sidebar, fill in:
- MySQL host / port / user / password / database name
- Groq API key

Click **Connect**, then start asking questions like:

- "How many customers do we have?"
- "Show me the top 5 products by total sales"
- "What was total revenue last month?"

## How it works

1. **`create_sql_query_chain`** (LangChain) takes your question + DB schema and
   asks the LLM (Llama 3.3 70B via Groq) to generate a MySQL query.
2. **`QuerySQLDataBaseTool`** executes that query against your real database.
3. A second LLM call turns the raw SQL result into a readable answer.
4. The UI shows the answer, and lets you expand to see the exact SQL that ran
   and the raw result — useful for debugging/trust.

## Notes / gotchas

- The LLM only sees your **schema** (table/column names), not your data, when
  writing the query — so it can occasionally guess a wrong column name on
  ambiguous schemas. If that happens, you can pass `db.get_table_info()` a
  `sample_rows_in_table_info=2` param inside `SQLDatabase(...)` in `app.py` to
  give it a few sample rows as extra context.
- Read-only use is safest. If you want to hard-block `INSERT/UPDATE/DELETE`,
  create a MySQL user with **SELECT-only** grants and use those credentials
  here instead of root.
- If you hit a `cryptography` / auth-plugin error connecting to MySQL 8+,
  it's already included in requirements.txt — that fixes the common
  `caching_sha2_password` issue with PyMySQL.
