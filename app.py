import json
import pandas as pd
from uuid import uuid4
from typing import Dict, Any, Optional, TypedDict, ClassVar
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import BaseTool
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langgraph.graph import StateGraph, END


load_dotenv()


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

            try:
                df = pd.read_csv(file_path)
                dataframes[df_name] = df
                print(f"Loaded: {df_name}  →  {df.shape[0]} rows")

            except Exception as e:
                print(f"Failed to load {file_path}: {e}")

    return dataframes


# Load everything dynamically
DATAFRAMES = load_csv_dataframes(DATA_FOLDER)



# schema for LLM reasoning
SCHEMAS = {
    name: list(df.columns)
    for name, df in DATAFRAMES.items()
}


vectorstore = Milvus(
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
    connection_args={"uri": "./milvus_multi.db", "db_name": "demo"},
    index_params={"index_type": "FLAT", "metric_type": "L2"},
    drop_old=True,
)

def ingest_df():
    docs = []
    for name, df in DATAFRAMES.items():
        for _, row in df.iterrows():
            text = f"{name} | " + " | ".join(f"{col}: {row[col]}" for col in df.columns)
            docs.append(Document(page_content=text))

    vectorstore.add_documents(docs, ids=[str(uuid4()) for _ in docs])
    print("Ingested into Milvus:", len(docs))

ingest_df()



class GreetingTool(BaseTool):
    name: ClassVar[str] = "greeting"
    description: ClassVar[str] = "Responds to simple greetings."

    def _run(self, query: str) -> str:
        import random
        return random.choice([
            "Hello! How can I help you?",
            "Hey there! What can I do for you?",
            "Hi! How may I assist you today?"
        ])


class RAGTool(BaseTool):
    name: ClassVar[str] = "rag"
    description: ClassVar[str] = "Retrieves contextual information from Milvus."

    def _run(self, query: str) -> str:
        try:
            docs = vectorstore.similarity_search(query, k=10)
            context = "\n".join(d.page_content for d in docs)

            prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question: {query}
Keep the answer concise and factual.
"""
            return llm.invoke(prompt).content
        except Exception as e:
            return f"[RAG error] {e}"



from typing import ClassVar

class CodeExecutionTool(BaseTool):
    name: ClassVar[str] = "code_executor"
    description: ClassVar[str] = "Executes Python code for analytical queries using multiple dataframes with auto-retry."

    MAX_RETRIES: ClassVar[int] = 3

    def _run(self, query: str) -> str:

        df_info = "\n".join(f"{name}: {cols}" for name, cols in SCHEMAS.items())

        base_prompt = f"""
You are an expert Python data engineer.

AVAILABLE DATAFRAMES:
{df_info}

RULES:
- Use ONLY these dataframes: {list(DATAFRAMES.keys())}
- Use ONLY pandas code.
- DO NOT import anything.
- Final result MUST be assigned to `output`.
- Only return python code, no markdown.
"""

        last_error = None

        for attempt in range(self.MAX_RETRIES):

            prompt = base_prompt + f"\nAttempt: {attempt+1}\nUser query: {query}"

            generated_code = llm.invoke(prompt).content
            generated_code = (
                generated_code.replace("```python", "")
                .replace("```", "")
                .strip()
            )

            local_env = {"output": None}
            local_env.update(DATAFRAMES)

            try:
                exec(generated_code, {}, local_env)
                return f"""
Generated Code (SUCCESS on attempt {attempt+1}):
--------------------------------
{generated_code}

Execution Output:
--------------------------------
{local_env['output']}
                """.strip()

            except Exception as e:
                last_error = e
                query = f"""
The previous code caused this error:
{e}

Fix your code.
                """

        return f"""
FAILED after {self.MAX_RETRIES} attempts.

Last generated code:
--------------------
{generated_code}

Last Error:
--------------------
{last_error}
""".strip()




TOOLS = {
    "greeting": GreetingTool(),
    "rag": RAGTool(),
    "code_executor": CodeExecutionTool(),
}



class AgentState(TypedDict):
    query: str
    next: Optional[str]
    result: Optional[str]


def router(state: AgentState) -> Dict[str, Any]:
    q = state["query"].lower()

    # greetings
    if q.startswith(("hi", "hello", "hey", "good morning", "good evening")):
        return {"next": "greeting"}

    # analytical => CodeExecutionTool
    analytic_keywords = ["how many", "total", "count", "sum", "average",
                         "filter", "top", "group", "year", "month",
                         "sold", "sales", "2022", "2023"]

    if any(k in q for k in analytic_keywords):
        return {"next": "code_executor"}

    # fallback: RAG
    return {"next": "rag"}


graph = StateGraph(AgentState)

graph.add_node("router", router)
graph.add_node("greeting", lambda s: {"result": TOOLS["greeting"].invoke({"query": s["query"]})})
graph.add_node("rag", lambda s: {"result": TOOLS["rag"].invoke({"query": s["query"]})})
graph.add_node("code_executor", lambda s: {"result": TOOLS["code_executor"].invoke({"query": s["query"]})})

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    lambda s: s["next"],
    {
        "greeting": "greeting",
        "rag": "rag",
        "code_executor": "code_executor",
    },
)

graph.add_edge("greeting", END)
graph.add_edge("rag", END)
graph.add_edge("code_executor", END)

app = graph.compile()


print("\nMulti-DF LangGraph Agent Ready!")

while True:
    q = input("\nYou: ")
    if q.lower() == "exit":
        break

    state = {"query": q, "next": None, "result": None}
    out = app.invoke(state)

    print("\nAgent Output:")
    print(out["result"])
