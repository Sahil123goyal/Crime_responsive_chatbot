"""
Text-to-SQL Chatbot for MySQL
------------------------------
Ask questions in plain English, get SQL generated + executed against your
MySQL database, with results shown in a chat UI.

Stack: Streamlit + LangChain + Groq (Llama 3.3 70B) + SQLAlchemy/PyMySQL
"""

import os
import re
import streamlit as st
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_classic.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

st.set_page_config(page_title="Text-to-SQL Chatbot", page_icon="🗄️", layout="wide")


def clean_sql(raw: str) -> str:
    """Strip markdown fences, 'SQLQuery:' prefixes, and prompt echoes some
    models (esp. Groq/Llama) add around the actual SQL. Raises a clear
    error instead of ever sending non-SQL text to the database."""
    text = raw.strip()

    # If the model echoed "SQLQuery:" (or similar), keep only what's after it
    match = re.search(r"SQLQuery:\s*", text, flags=re.IGNORECASE)
    if match:
        text = text[match.end():]

    # Strip markdown code fences (```sql ... ``` or ``` ... ```)
    text = re.sub(r"```sql", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")

    # Strip markdown bold markers some models (e.g. gpt-oss) wrap the query
    # in instead of code fences (**SELECT ...**). Only strip double-asterisk
    # bold markers — a single "*" is the legitimate SQL wildcard (SELECT *).
    text = text.replace("**", "")

    # If there's still a "Question:" style echo before the query, cut it
    text = re.sub(r"^.*?Question:.*?(SELECT|WITH|INSERT|UPDATE|DELETE|SHOW|DESCRIBE)",
                   r"\1", text, flags=re.IGNORECASE | re.DOTALL)

    text = text.strip()

    # Cut off at the first semicolon: some models append trailing
    # commentary/echoes ("**SQL Query:**", explanations, etc.) after the
    # actual query ends. We only want the first statement.
    if ";" in text:
        text = text.split(";", 1)[0]

    text = " ".join(text.split()).strip().rstrip(";").strip()

    # Validate: must actually start with a SQL command. If the model
    # replied with an apology / clarifying question instead of SQL,
    # fail loudly here rather than executing it against MySQL.
    if not re.match(r"^(SELECT|WITH|INSERT|UPDATE|DELETE|SHOW|DESCRIBE)\b", text, flags=re.IGNORECASE):
        shown = raw.strip() or "(empty response)"
        raise ValueError(
            "The model didn't return a valid SQL query for that question "
            f"(it said: \u201c{shown[:150]}\u2026\u201d). Try rephrasing your "
            "question to be more specific, e.g. mention the table or column name."
        )

    return text + ";"


# Common greetings / small talk that shouldn't be sent through the SQL
# pipeline at all. Kept intentionally simple (no LLM call) so it's fast,
# free, and predictable. Anything not matched here still goes through the
# normal SQL flow and clean_sql's validation as a safety net.
_SMALL_TALK_RE = re.compile(
    r"^\s*(hi|hello|hey|yo|sup|good\s?(morning|afternoon|evening)|"
    r"how('?s| is| are) it going|how are you|what'?s up|"
    r"thanks|thank you|thx|ty|bye|goodbye|see ya|ok|okay|cool|nice|great)"
    r"[\s!.,?]*$",
    flags=re.IGNORECASE,
)


def small_talk_reply(text: str) -> str | None:
    """Return a friendly canned reply if the message looks like small talk
    rather than a database question, else None."""
    if not _SMALL_TALK_RE.match(text.strip()):
        return None
    lowered = text.strip().lower()
    if lowered.startswith(("thanks", "thank you", "thx", "ty")):
        return "You're welcome! Ask me anything about your database whenever you're ready. 🙂"
    if lowered.startswith(("bye", "goodbye", "see ya")):
        return "Goodbye! Come back anytime you want to query your database. 👋"
    return "Hey there! 👋 I'm your database assistant — ask me a question about your data, e.g. \"What are the top 5 customers by total revenue?\""

# ----------------------------
# Sidebar: DB + API connection
# ----------------------------
st.sidebar.title("⚙️ Connection Settings")

db_host = st.sidebar.text_input("MySQL Host", value=os.getenv("DB_HOST", "localhost"))
db_port = st.sidebar.text_input("Port", value=os.getenv("DB_PORT", "3306"))
db_user = st.sidebar.text_input("User", value=os.getenv("DB_USER", "root"))
db_password = st.sidebar.text_input("Password", value=os.getenv("DB_PASSWORD", ""), type="password")
db_name = st.sidebar.text_input("Database", value=os.getenv("DB_NAME", ""))

groq_api_key = st.sidebar.text_input(
    "Groq API Key", value=os.getenv("GROQ_API_KEY", ""), type="password"
)

connect_btn = st.sidebar.button("🔌 Connect")

if "db" not in st.session_state:
    st.session_state.db = None
if "answer_chain" not in st.session_state:
    st.session_state.answer_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# Connect to DB + build chain
# ----------------------------
if connect_btn:
    try:
        uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        engine = create_engine(uri)
        db = SQLDatabase(engine)

        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
            api_key=groq_api_key,
        )

        # Chain 1: NL question -> SQL query
        write_query = create_sql_query_chain(llm, db)

        # Chain 2: execute SQL query against DB
        execute_query = QuerySQLDataBaseTool(db=db)

        # Chain 3: prompt used to turn (question, sql, result) into a
        # natural-language answer. Invoked manually below, using the SAME
        # sql_query/result we already computed — avoids a second, possibly
        # inconsistent, LLM call to regenerate the SQL.
        answer_prompt = PromptTemplate.from_template(
            """You are a helpful data analyst. Given the user's question, the SQL query
that was run, and the SQL result, write a clear, concise natural-language answer.
If the result is a table, summarize the key numbers/rows instead of dumping everything.

Question: {question}
SQL Query: {query}
SQL Result: {result}

Answer:"""
        )
        answer_chain = answer_prompt | llm | StrOutputParser()

        st.session_state.db = db
        st.session_state.write_query = write_query
        st.session_state.execute_query = execute_query
        st.session_state.answer_chain = answer_chain
        st.sidebar.success(f"Connected to `{db_name}` ✅")

        with st.sidebar.expander("📋 Detected tables"):
            st.write(db.get_usable_table_names())

    except Exception as e:
        st.sidebar.error(f"Connection failed: {e}")

st.title("🗄️ Chat with your MySQL Database")
st.caption("Ask questions in plain English — I'll write and run the SQL for you.")

if st.session_state.answer_chain is None:
    st.info("👈 Fill in your MySQL + Groq credentials in the sidebar and click **Connect** to start.")
else:
    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sql" in msg:
                with st.expander("🔍 Generated SQL"):
                    st.code(msg["sql"], language="sql")

    user_q = st.chat_input("e.g. What are the top 5 customers by total revenue?")

    if user_q:
        st.session_state.messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            small_talk = small_talk_reply(user_q)
            if small_talk is not None:
                st.markdown(small_talk)
                st.session_state.messages.append({"role": "assistant", "content": small_talk})
            else:
                with st.spinner("Writing SQL and querying database..."):
                    try:
                        sql_query = clean_sql(
                            st.session_state.write_query.invoke({"question": user_q})
                        )
                        sql_result = st.session_state.execute_query.invoke(sql_query)
                        answer = st.session_state.answer_chain.invoke(
                            {"question": user_q, "query": sql_query, "result": sql_result}
                        )

                        st.markdown(answer)
                        with st.expander("🔍 Generated SQL"):
                            st.code(sql_query, language="sql")
                        with st.expander("📊 Raw result"):
                            st.text(sql_result)

                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer, "sql": sql_query}
                        )
                    except Exception as e:
                        err = f"⚠️ Something went wrong: {e}"
                        st.error(err)
                        st.session_state.messages.append({"role": "assistant", "content": err})