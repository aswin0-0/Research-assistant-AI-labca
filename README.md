# Research Assistant Agent 🔍
### LangChain + Groq (FREE) | Wikipedia + Calculator tools

---

## What This Is
A **ReAct-style LangChain agent** that acts as a Research Assistant.
Given a user question, it autonomously:
1. Thinks about what it needs
2. Searches Wikipedia for facts
3. Uses a Calculator for any numbers
4. Loops until it can give a complete answer

---

## FREE API Setup (No credit card needed)

### Step 1 — Get your free Groq API key
1. Go to → **https://console.groq.com**
2. Sign up with email (free, no card)
3. Click **"API Keys"** → **"Create API Key"**
4. Copy the key (starts with `gsk_...`)

### Step 2 — Set the key
**Option A — Environment variable (recommended)**
```bash
# Linux / Mac
export GROQ_API_KEY="gsk_your_key_here"

# Windows CMD
set GROQ_API_KEY=gsk_your_key_here
```

**Option B — Edit agent.py directly**
```python
GROQ_API_KEY = "gsk_your_key_here"   # line 19 in agent.py
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the Agent

```bash
python agent.py
```

To ask a custom question, edit the bottom of `agent.py`:
```python
run_research_agent("Your research question here")
```

---

## Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────┐
│         ReAct Agent (LangChain)     │
│  LLM: Llama-3.3-70b (Groq FREE)    │
│                                     │
│  Loop:                              │
│    Thought → Action → Observation   │
│         ↑________________________|  │
└───────────────┬─────────────────────┘
                │ calls
       ┌────────┴────────┐
       ▼                 ▼
  [Wikipedia Tool]  [Calculator Tool]
  (fetches facts)   (computes numbers)
       │                 │
       └────────┬────────┘
                ▼
          Final Answer
```

---

## Tools Explained

| Tool | Source | Purpose |
|------|--------|---------|
| Wikipedia | `langchain_community` + `wikipedia` pkg | Factual research, background knowledge |
| Calculator | `LLMMathChain` (LangChain built-in) | Any numeric computation during research |

---

## Agent Type: ReAct
**ReAct = Reasoning + Acting**  
The agent alternates between:
- **Thought** — what does it need to find out?
- **Action** — which tool to call and with what input
- **Observation** — what did the tool return?

This loop repeats until the agent has enough to answer.

---

## Free Tier Limits (Groq)
- ~14,400 requests/day on free tier
- Rate limit: 30 requests/minute
- More than enough for this assignment
