"""
Research Assistant Agent
========================
Framework : LangGraph prebuilt ReAct agent (LangGraph 1.0+)
LLM       : Llama-3.3-70b-versatile via Groq (FREE tier)
Tools     : 1) Wikipedia Search  2) Calculator
"""

import os
import math
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.prebuilt import create_react_agent

# ── 0. Load API Key ───────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── 1. LLM ────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0,
)

# ── 2. Tool 1 – Wikipedia ─────────────────────────────────────────────────────
wiki_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1500)

@tool("wikipedia_search")
def wikipedia(query: str) -> str:
    """Search Wikipedia for factual information, definitions, historical facts,
    or scientific concepts. Input: a plain search query string."""
    try:
        result = wiki_wrapper.run(query)
        return result if result else "No Wikipedia article found."
    except Exception as e:
        return f"Wikipedia error: {e}"

# ── 3. Tool 2 – Calculator ────────────────────────────────────────────────────
@tool("calculator")
def calculator(expression: str) -> str:
    """Evaluate a math expression. Input MUST be numeric only.
    Examples: '149600000 / 299792.458', '3.7 * 0.05', 'sqrt(144)'.
    Never pass English sentences — only numbers and operators."""
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed["abs"] = abs
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

# ── 4. Build Agent ────────────────────────────────────────────────────────────
tools = [wikipedia, calculator]

# Explicitly bind tools to LLM — fixes Groq tool name mismatch
llm_with_tools = llm.bind_tools(tools)

system_message = SystemMessage(content=(
    "You are a Research Assistant Agent. "
    "Use wikipedia_search to look up facts and calculator for any math. "
    "Always gather facts first, then calculate if needed. "
    "For calculator, pass ONLY numeric expressions like '149600000 / 299792.458'."
))

agent = create_react_agent(
    model=llm_with_tools,
    tools=tools,
    prompt=system_message,
)

# ── 5. Runner ─────────────────────────────────────────────────────────────────
def run_research_agent(question: str) -> str:
    print("\n" + "=" * 70)
    print(f"  USER GOAL : {question}")
    print("=" * 70 + "\n")

    final_answer = ""
    for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        msg = step["messages"][-1]
        msg.pretty_print()
        final_answer = msg.content

    print("\n" + "=" * 70)
    print("  FINAL ANSWER")
    print("=" * 70)
    print(final_answer)
    return final_answer

# ── 6. Interactive Loop ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔍 Research Assistant Agent (type 'exit' to quit)\n")
    while True:
        question = input("Ask a research question: ").strip()
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if not question:
            continue
        run_research_agent(question)