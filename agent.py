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
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.prebuilt import create_react_agent

# ── 0. Load API Key ───────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── 1. LLM — pass raw, no bind_tools ─────────────────────────────────────────
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
)

# ── 2. Tool 1 – Wikipedia ─────────────────────────────────────────────────────
wiki_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1500)

@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for factual information, definitions, historical facts,
    or scientific concepts. Input: a plain search query string."""
    try:
        result = wiki_wrapper.run(query)
        return result if result else "No Wikipedia article found."
    except Exception as e:
        return f"Wikipedia error: {e}"

# ── 3. Tool 2 – Calculator ────────────────────────────────────────────────────
@tool
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

# ── 4. Build Agent — let LangGraph handle tool binding ────────────────────────
tools = [wikipedia_search, calculator]

system_message = SystemMessage(content=(
    "You are a Research Assistant Agent. "
    "Always use the wikipedia_search tool to find facts before answering. "
    "Use the calculator tool for any numeric computation. "
    "Think step by step. Explain your reasoning before each action."
))

agent = create_react_agent(       # LangGraph handles bind_tools internally
    model=llm,
    tools=tools,
    prompt=system_message,
)

# ── 5. Pretty Step Printer ────────────────────────────────────────────────────
def print_step(msg, step_num):
    if isinstance(msg, HumanMessage):
        print(f"\n{'─'*70}")
        print(f"  📥 USER QUESTION")
        print(f"{'─'*70}")
        print(f"  {msg.content}")

    elif isinstance(msg, AIMessage):
        if msg.tool_calls:
            thought_text = msg.content.strip() if msg.content else ""
            tool_call    = msg.tool_calls[0]
            tool_name    = tool_call["name"]
            tool_input   = tool_call["args"]

            print(f"\n{'─'*70}")
            print(f"  🧠 STEP {step_num} — AGENT REASONING")
            print(f"{'─'*70}")

            if thought_text:
                print(f"  Thought : {thought_text}")
            else:
                if tool_name == "wikipedia_search":
                    query = tool_input.get("query", "")
                    print(f"  Thought : I need to find information about '{query}'. "
                          f"I will search Wikipedia to gather relevant facts before answering.")
                elif tool_name == "calculator":
                    expr = tool_input.get("expression", "")
                    print(f"  Thought : I now have the facts I need. "
                          f"I need to calculate '{expr}' to produce the final numeric answer.")

            print(f"\n  Action  : {tool_name}")
            print(f"  Input   : {tool_input}")

        elif msg.content:
            print(f"\n{'─'*70}")
            print(f"  ✅ AGENT FINAL ANSWER")
            print(f"{'─'*70}")
            print(f"  {msg.content}")

    elif isinstance(msg, ToolMessage):
        content = msg.content
        if len(content) > 400:
            content = content[:400] + "...[truncated]"
        print(f"\n  📋 OBSERVATION (from {msg.name})")
        print(f"{'─'*70}")
        print(f"  {content}")
        print(f"\n  ➡  Agent will now reason about this and decide the next step...")

# ── 6. Runner ─────────────────────────────────────────────────────────────────
def run_research_agent(question: str) -> str:
    print("\n" + "=" * 70)
    print(f"  USER GOAL : {question}")
    print("=" * 70)

    step_num    = 1
    final_answer = ""
    seen_ids    = set()

    for state in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        msg    = state["messages"][-1]
        msg_id = getattr(msg, "id", None)
        if msg_id and msg_id in seen_ids:
            continue
        if msg_id:
            seen_ids.add(msg_id)

        print_step(msg, step_num)

        if isinstance(msg, AIMessage) and msg.tool_calls:
            step_num += 1
        if isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
            final_answer = msg.content

    print("\n" + "=" * 70)
    print("  FINAL ANSWER")
    print("=" * 70)
    print(final_answer)
    return final_answer

# ── 7. Interactive Loop ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔍 Research Assistant Agent")
    print("   Tools : Wikipedia Search + Calculator")
    print("   Type 'exit' to quit\n")
    while True:
        question = input("Ask a research question: ").strip()
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if not question:
            continue
        run_research_agent(question)