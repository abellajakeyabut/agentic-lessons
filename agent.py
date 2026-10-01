from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph
from typing import TypedDict

class AgentState(TypedDict):
    input: str
    output: str
    summary: str
    factcheck: str
    sentiment: str
    style: str

llm = ChatOllama(model="codellama", base_url="http://localhost:11434")

def summarizer_node(state: AgentState) -> AgentState:
    summary = llm.invoke(f"Summarize the following text:\n{state['input']}").content
    return {**state, "summary": summary}

def sentiment_analyzer(state: AgentState) -> AgentState:
    sentiment = llm.invoke(f"Analyze the sentiment:\n{state['summary']}").content
    return {"sentiment": sentiment}

def fact_checker(state: AgentState) -> AgentState:
    fact_check = llm.invoke(f"Check the facts:\n{state['summary']}").content
    return { "factcheck": fact_check}

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

graph.set_entry_point("summarizer")
graph.add_edge("summarizer", "sentiment_analyzer")
graph.add_edge("summarizer", "fact_checker")
graph.add_edge("fact_checker", "stylistic_writer")
graph.add_edge("stylistic_writer", "aggregator")

app = graph.compile()

if __name__ == "__main__":
    text = "The quick brown fox jumps over the lazy dog"
    result = app.invoke({"input": text})
    print(result["output"])
