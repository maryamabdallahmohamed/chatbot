from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from dotenv import load_dotenv
from typing import Annotated, Literal
from pydantic import BaseModel
from typing_extensions import TypedDict
import os


load_dotenv()
gemini_key = os.getenv("gemini_key")
tavily_key = os.getenv("TAVILY")


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", api_key=gemini_key)
tavily_client = TavilyClient(api_key=tavily_key)
MAX_HISTORY = 5


class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str


class MessageClassifier(BaseModel):
    message_type: Literal["logical", "online_search"]


graph_builder = StateGraph(State)

# --- NODES ---

# Classifier Node
def classify_message(state: State):
    last_message = state["messages"][-1].content
    result = llm.with_structured_output(MessageClassifier).invoke(
        [{"role": "user", "content": f"Classify if this message needs logical reasoning or online search: {last_message}"}]
    )
    return {"message_type": result.message_type, "messages": state["messages"][-MAX_HISTORY:]}

# Chatbot Node (logical reasoning)
def chatbot(state: State):
    response = llm.invoke(state["messages"][-MAX_HISTORY:])
    return {"messages": state["messages"] + [response]}

# Search Node (fact-check / online search)
def search_online(state: State):
    query = state["messages"][-1].content
    search_results = tavily_client.search(
        query=query,
        search_depth="advanced",
        include_domains=[".org", ".edu", ".eg", ".gov"],
        include_answer=True,
        max_results=3
    )
    return {"messages": state["messages"] + [{"role": "assistant", "content": str(search_results)}]}

# --- BUILD GRAPH ---
graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("search_online", search_online)

graph_builder.add_edge(START, "classifier")

# Router Node
def router_node(state: State):
    if state["message_type"] == "logical":
        return "chatbot"
    else:
        return "search_online"

graph_builder.add_conditional_edges("classifier", router_node, {"chatbot": "chatbot", "search_online": "search_online"})
graph_builder.add_edge("chatbot", END)
graph_builder.add_edge("search_online", END)

graph = graph_builder.compile()

# --- TEST RUN ---
if __name__ == "__main__":
    user_input = "WHich egyptian nationalists won in the last olympics?"
    result = graph.invoke({"messages": [{"role": "user", "content": user_input}]})
    print("🤖 Response:", result["messages"][-1])

