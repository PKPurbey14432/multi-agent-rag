

import streamlit as st
import pandas as pd
import json
from uuid import uuid4
from typing import Dict, Any, Optional, TypedDict, ClassVar
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import BaseTool
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()


st.set_page_config(page_title="Multi-DF Agent", layout="wide")
st.title("Multi-Agent Chat-Bot")
st.caption("RAG + Code Execution + Smart Routing")

# persist history
if "history" not in st.session_state:
    st.session_state.history = []  # list of (role, text)

llm = ChatOpenAI(model="gpt-4.1", temperature=0)



DATA_FOLDER = os.getenv("DATA_FOLDER", "./data")

def normalize_df_name(filename: str) -> str:
    """
    Convert filename into a clean python-safe variable name.

    Example:
    "Amazon Sale Report.csv" -> "Amazon_Sale_Report"
    "P  L March 2021.csv" -> "P_L_March_2021"
    "May-2022.csv" -> "May_2022"
    """
    name = filename.replace(".csv", "")
    name = name.replace("-", "_").replace(" ", "_")
    while "__" in name:
        name = name.replace("__", "_")
    return name


def load_csv_dataframes(folder: str):
    dataframes = {}

    for file in os.listdir(folder):
        if file.lower().endswith(".csv"):
            file_path = os.path.join(folder, file)
            df_name = normalize_df_name(file)
            print(f"Loading: {df_name}  →  {file_path}")
            try:
                df = pd.read_csv(file_path)
                dataframes[df_name] = df
                print(f"Loaded: {df_name}  →  {df.shape[0]} rows")

            except Exception as e:
                st.error(f"Failed to load {file_path}: {e}")

    return dataframes


# Load everything dynamically
DATAFRAMES = load_csv_dataframes(DATA_FOLDER)


# Create schema map dynamically
SCHEMAS = {name: list(df.columns) for name, df in DATAFRAMES.items()}


vectorstore = Milvus(
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
    connection_args={"uri": "./milvus_multi.db", "db_name": "demo"},
    index_params={"index_type": "FLAT", "metric_type": "L2"},
    drop_old=True,
)

def ingest_sample():
    docs = []
    for name, df in DATAFRAMES.items():
        sample = df.copy()
        for idx, row in sample.iterrows():
            text = f"{name} | index: {idx} | " + " | ".join(f"{col}: {row[col]}" for col in df.columns)
            docs.append(Document(page_content=text))
    if docs:
        vectorstore.add_documents(docs, ids=[str(uuid4()) for _ in docs])
        st.write(f"[Milvus] Ingested {len(docs)} sample docs.")

# ingest a small sample at startup (adjust sample_per_df as needed)
ingest_sample()


class GreetingTool(BaseTool):
    name: ClassVar[str] = "greeting"
    description: ClassVar[str] = "Simple greeting responses."

    def _run(self, query: str) -> str:
        import random
        return random.choice([
            "Hello! 😊 How can I help you?",
            "Hey there! What can I do for you?",
            "Hi! How may I assist you today?"
        ])


class RAGTool(BaseTool):
    name: ClassVar[str] = "rag"
    description: ClassVar[str] = "Return RAG result or __NO_ANSWER__ sentinel."

    def _run(self, query: str) -> str:
        try:
            docs = vectorstore.similarity_search(query, k=10)
            if not docs:
                return "__NO_ANSWER__"
            context = "\n".join(d.page_content for d in docs)
            # instruct LLM to return __NO_ANSWER__ if context doesn't contain the answer
            prompt = f"""
Use ONLY the context below to answer the question.
If the answer is not contained in the context, reply EXACTLY with "__NO_ANSWER__".

CONTEXT:
{context}

QUESTION:
{query}
"""
            ans = llm.invoke(prompt).content.strip()
            if not ans:
                return "__NO_ANSWER__"
            return ans
        except Exception:
            return "__NO_ANSWER__"


class CodeExecutionTool(BaseTool):
    name: ClassVar[str] = "code_executor"
    description: ClassVar[str] = "Generates and executes pandas code (retries), returns only final result."
    MAX_RETRIES: ClassVar[int] = 3

    def _run(self, query: str) -> str:
        # provide schema context to LLM
        df_info = "\n".join(f"{name}: {cols}" for name, cols in SCHEMAS.items())
        base_prompt = f"""
You are an expert Python data engineer.

AVAILABLE DATAFRAMES (name: columns):
{df_info}

RULES:
- Use ONLY the DataFrame names above.
- Do NOT import anything.
- Use pandas operations only.
- The final answer MUST be assigned to variable `output`.
- Return ONLY valid python code (no markdown, no explanation).
User question:
{query}
"""
        last_error = None
        generated_code = ""
        for attempt in range(self.MAX_RETRIES):
            # ask LLM for code
            generated_code = llm.invoke(base_prompt).content
            generated_code = generated_code.replace("```python", "").replace("```", "").strip()

            # prepare safe env
            local_env = {"output": None}
            local_env.update(DATAFRAMES)

            try:
                exec(generated_code, {}, local_env)
                result = local_env.get("output")
                # If DataFrame or Series, return JSON string (split orient)
                if isinstance(result, (pd.DataFrame, pd.Series)):
                    try:
                        return result.to_json(orient="split", force_ascii=False)
                    except Exception:
                        return str(result)
                return str(result)
            except Exception as e:
                last_error = e
                # update prompt to include error and request corrected code next attempt
                base_prompt += f"\n\nPrevious attempt raised: {e}\nPlease return corrected code only."
                continue
        # failed all retries
        return f"__CODE_ERROR__: {last_error}\nLastCode:\n{generated_code}"


TOOLS = {
    "greeting": GreetingTool(),
    "rag": RAGTool(),
    "code_executor": CodeExecutionTool(),
}


class AgentState(TypedDict):
    query: str
    next: Optional[str]
    rag_answer: Optional[str]
    result: Optional[str]


def router(state: AgentState) -> Dict[str, Any]:
    q = state["query"].strip()
    ql = q.lower()

    # greeting detection
    if ql.startswith(("hi", "hello", "hey", "good morning", "good afternoon", "good evening")):
        return {"query": q, "next": "greeting", "rag_answer": None, "result": None}

    # run RAG right away and store rag_answer in state
    rag_out = TOOLS["rag"].invoke({"query": q})
    if isinstance(rag_out, str) and rag_out.strip() == "__NO_ANSWER__":
        # no answer in vector DB -> go to code_executor
        return {"query": q, "next": "code_executor", "rag_answer": None, "result": None}
    # RAG found something -> go to final and carry rag_answer
    return {"query": q, "next": "final", "rag_answer": rag_out, "result": None}


def greeting_node(state: AgentState) -> Dict[str, Any]:
    q = state["query"]
    out = TOOLS["greeting"].invoke({"query": q})
    return {"query": q, "next": None, "rag_answer": None, "result": out}


def code_executor_node(state: AgentState) -> Dict[str, Any]:
    q = state["query"]
    out = TOOLS["code_executor"].invoke({"query": q})
    # store result so final_node can access it
    return {"query": q, "next": "final", "rag_answer": None, "result": out}


def final_node(state: AgentState) -> Dict[str, Any]:
    q = state["query"]
    rag = state.get("rag_answer")
    code_out = state.get("result")

    # If rag present -> return it exactly as-is
    if rag:
        return {"query": q, "next": None, "rag_answer": rag, "result": rag}

    # If code execution produced JSON for df -> convert to table preview
    if isinstance(code_out, str) and code_out.startswith("{") and "columns" in code_out and "data" in code_out:
        try:
            parsed = json.loads(code_out)
            if isinstance(parsed, dict) and "columns" in parsed and "data" in parsed:
                df_preview = pd.DataFrame(parsed["data"], columns=parsed["columns"])
                # return markdown table preview
                return {"query": q, "next": None, "rag_answer": None, "result": df_preview.to_markdown()}
        except Exception:
            pass

    # If code execution returned special error sentinel
    if isinstance(code_out, str) and code_out.startswith("__CODE_ERROR__"):
        return {"query": q, "next": None, "rag_answer": None, "result": "I attempted to compute the answer but failed to execute the generated code."}

    # Otherwise ask LLM to convert raw code result into a short readable answer
    if code_out is not None:
        try:
            prompt = f"Convert this raw result into a short, human-readable answer:\n\n{code_out}"
            formatted = llm.invoke(prompt).content.strip()
            return {"query": q, "next": None, "rag_answer": None, "result": formatted}
        except Exception:
            return {"query": q, "next": None, "rag_answer": None, "result": str(code_out)}

    # fallback
    return {"query": q, "next": None, "rag_answer": None, "result": "No answer found."}


graph = StateGraph(AgentState)

graph.add_node("router", router)
graph.add_node("greeting", greeting_node)
graph.add_node("code_executor", code_executor_node)
graph.add_node("final", final_node)

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    lambda s: s["next"],
    {
        "greeting": "greeting",
        "code_executor": "code_executor",
        "final": "final",
    },
)

# transitions
graph.add_edge("greeting", END)
graph.add_edge("code_executor", "final")  
graph.add_edge("final", END)

app = graph.compile()


def run_agent(user_input: str) -> str:
    # ensure AgentState keys exist
    state = {"query": user_input, "next": None, "rag_answer": None, "result": None}
    out = app.invoke(state)
    # out should contain 'result'
    return out.get("result", "No result")


st.subheader("Chat")

# render history
for role, text in st.session_state.history:
    with st.chat_message(role):
        st.write(text)

user_input = st.chat_input("Ask something...")
if user_input:
    # append and immediately render user message
    st.session_state.history.append(("user", user_input))
    with st.chat_message("user"):
        st.write(user_input)

    # run and show assistant answer
    with st.chat_message("assistant"):
        answer = run_agent(user_input)
        st.write(answer)
    st.session_state.history.append(("assistant", answer))
