from langchain_ollama import OllamaEmbeddings, ChatOllama
from langgraph.graph import StateGraph
from typing import TypedDict

from langchain_community.vectorstores import Chroma;
from langchain_community.embeddings import OllamaEmbeddings;


class AgentState(TypedDict):
    input: str
    output: str
    summary: str
    factcheck: str
    sentiment: str
    style: str
    retrieved:str

llm = ChatOllama(model="codellama", base_url="http://localhost:11434")
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
vectorstore = Chroma(embedding_function=embeddings,collection_name="my_docs" ,persist_directory="./chroma_db");

def retriever_node(state: AgentState) -> AgentState:
    docs = vectorstore.similarity_search(state['summary'], k=3)
    retrieved_text = "\n".join([doc.page_content for doc in docs])
    return {**state, "retrieved": retrieved_text}

def fact_checker(state: AgentState) -> AgentState:
    fact_check = llm.invoke(f"Check for the following summary against these sources:\n\n"
                            f"Summary:\n{state['summary']}\n\n"
                            f"Sources:\n{state['retrieved']}").content

    return {**state, "factcheck": fact_check}

def summarizer_node(state: AgentState) -> AgentState:
    summary = llm.invoke(f"Summarize the following text:\n{state['input']}").content
    return {**state, "summary": summary}

def sentiment_analyzer(state: AgentState) -> AgentState:
    sentiment = llm.invoke(f"Analyze the sentiment:\n{state['summary']}").content
    return {"sentiment": sentiment}

def stylistic_writer_node(state: AgentState) -> AgentState:
    styled = llm.invoke(f"Rewrite stylishly:\n{state['summary']}").content
    return { **state, "style": styled}

def aggregator_node(state: AgentState) -> AgentState:
    final_output = (
        f"Summary:\n{state['summary']}\n\n"
        f"Sentiment:\n{state['sentiment']}\n\n"
        f"Fact‑Check:\n{state['factcheck']}\n\n"
        f"Stylish Rewrite:\n{state['style']}"
    )
    return {**state, "output": final_output}

graph = StateGraph(AgentState)
graph.add_node("summarizer", summarizer_node)
graph.add_node("sentiment_analyzer", sentiment_analyzer)
graph.add_node("fact_checker", fact_checker)
graph.add_node("stylistic_writer", stylistic_writer_node)
graph.add_node("aggregator", aggregator_node)
graph.add_node("retriever", retriever_node)

graph.set_entry_point("summarizer")
graph.add_edge("summarizer", "sentiment_analyzer")
graph.add_edge("summarizer", "retriever")

graph.add_edge("retriever", "fact_checker")
graph.add_edge("fact_checker", "stylistic_writer")
graph.add_edge("stylistic_writer", "aggregator")

app = graph.compile()

if __name__ == "__main__":
    text = "The quick brown fox jumps over the lazy dog"
    result = app.invoke({"input": text})
    print(result["output"])
