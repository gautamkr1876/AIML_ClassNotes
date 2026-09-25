<a id="top"></a>
# Multi-Agent Systems with LangGraph — Jargon Card

> **How to use this file.** Skim it once (~6 min) before the notebook, then keep it open in a second tab while you read. Every AI/agent term the notebook uses has an entry here, in plain English first.
>
> **Companion:** read [`MultiAgent_LangGraph_Reading_Brief.md`](./MultiAgent_LangGraph_Reading_Brief.md) *first* — it gives you the punchline, the agenda, and the core concepts. This card is the dictionary you consult while reading.
>
> **Notebook:** `Realtime_langgraph_multiagent_travel_planner.ipynb` — a real-time travel planner built from 8 agents wired into one LangGraph graph.

---

## A

**Agent** — In this notebook, an agent is just **a plain Python function that takes the shared state and returns a partial update to it**. That's it: no class, no framework magic. Some agents call the LLM (supervisor, cost, itinerary, final), some only call a web API (flight, hotel, news, weather). The word "agent" here means "a specialist worker with one job", not "an autonomous entity that loops until done".

**Aviationstack** — A commercial flight-data API. The notebook's `flight_agent` sends it a departure airport code, an arrival code, and a date, and gets back a list of scheduled flights. The free tier is limited, which is why the notebook caps results at `limit=20`.

**API key** — A secret string that proves to an external service that you're allowed to call it. This notebook needs **five** of them (Gemini, Aviationstack, NewsData.io, Tavily, OpenWeather), all collected with `getpass` so they never appear in the notebook output.

## C

**Compile** — `builder.compile()` turns the graph *description* (nodes + edges you declared) into an executable `graph` object you can `.invoke()`. You must re-compile after changing any node or edge; forgetting to is a classic source of "my fix didn't take effect".

**Cost agent** — The one agent in this notebook that does **research plus synthesis**: it runs a live Tavily web search for prices, then feeds those search results *plus* the hotel and flight state into Gemini and asks for a structured JSON cost estimate. Its prompt says "Never fabricate a price" — that instruction is the guardrail.

## E

**Edge** — A one-way arrow between two nodes that says "after A finishes, B may run". `builder.add_edge("supervisor", "flight_agent")` is an edge. Edges are what encode the orchestration; the agents themselves know nothing about each other.

## F

**Fan-out** — One node branching into several that can run **independently**. Here the supervisor fans out into flight, hotel, news, and weather — four agents that share no data and could all run at once.

**Fan-in** — Several nodes converging on one. `cost_agent` fans in from flight + hotel; `itinerary_agent` fans in from five upstream nodes. Fan-in is where the subtle scheduling behaviour lives (see the Brief's gotchas).

**Final agent** — The last node. It doesn't fetch anything new — it re-reads everything in state and asks Gemini to write the polished, user-facing travel plan with sections for flights, hotels, weather, news, cost, and itinerary.

## G

**Gemini** — Google's family of large language models. The notebook wires it in via `ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)`. **Flash** models are the fast/cheap tier — the right choice when you're making many short calls.

**Geocoding** — Turning a place name into latitude/longitude numbers. The weather agent must geocode first ("Manali" → 32.2455, 77.1873) because OpenWeather's forecast endpoint only accepts coordinates, not names.

**Grounding** — Making the model answer from *supplied evidence* rather than from its own memory. Every LLM prompt in this notebook is grounded: it pastes the live API results into the prompt and adds rules like "Do not invent live facts" and "Do not claim a flight is bookable".

**getpass** — A Python function that prompts for a secret and hides the typing. Used so the five API keys never get saved into the `.ipynb` file.

## H

**Hallucination** — When a model states something confidently that isn't true. This is *the* risk in a travel planner (an invented hotel price or a non-existent flight). The countermeasure here is grounding plus explicit "never fabricate" rules in the prompts.

**Hotel agent** — Calls Tavily with `search_depth="advanced"` and `include_answer=True`, so it gets back both raw search results *and* a one-paragraph synthesized answer.

## I

**IATA code** — The three-letter airport/city code airlines use: Delhi = `DEL`, Goa = `GOI`. The supervisor extracts these from free text, but its prompt explicitly says **"Do not invent IATA codes"** — better an empty string than a wrong airport.

**Itinerary agent** — Reads *everything* gathered so far and asks Gemini for a day-by-day plan that is weather-aware and news-aware, with estimates clearly marked.

**invoke** — `graph.invoke({...})` runs the whole graph once, from `START` to `END`, and returns the final accumulated state dictionary.

## L

**LangChain** — The general-purpose library for talking to LLMs, wrapping tools, and formatting messages. Here it supplies `ChatGoogleGenerativeAI` and `HumanMessage`.

**LangGraph** — The orchestration library built on LangChain. Where LangChain gives you "call an LLM", LangGraph gives you "wire many steps into a graph with shared state, branches, and joins". It's the star of this notebook.

**LLM (Large Language Model)** — A model trained to predict the next chunk of text, which in practice means it can follow instructions, extract structure, and write prose. Gemini is the LLM here.

## M

**Mermaid diagram** — A text format for drawing diagrams. `graph.get_graph().draw_mermaid_png()` renders your compiled LangGraph as a picture — the fastest way to check you wired the edges you *meant* to wire.

**Multi-agent system** — Several specialised LLM/tool workers that together solve a task no single prompt would handle well. Compare **single-agent**: one prompt, one model, one shot. Multi-agent buys you specialisation, parallelism, and debuggability; it costs you orchestration complexity.

## N

**NewsData.io** — A news-search API. The `news_agent` queries it by destination name to surface anything that might disrupt travel (floods, strikes, events).

**Node** — A named step in the graph: `builder.add_node("hotel_agent", hotel_agent)` registers the Python function `hotel_agent` under the name `"hotel_agent"`. Node name and function name don't have to match, but keeping them identical saves confusion.

## O

**Orchestration** — Deciding *who runs when, with what, and in what order*. In this notebook orchestration lives entirely in the ~20 lines of `add_edge` calls — the agents themselves contain none of it.

**OpenWeather** — A weather API. Used in two steps: geocode the destination, then fetch current conditions in metric units.

## P

**Partial state update** — The key LangGraph convention: a node returns **only the keys it changed** (`return {"weather": {...}}`), and LangGraph merges that into the shared state. A node never rebuilds the whole state dictionary.

## R

**raise_for_status()** — A `requests` method that turns an HTTP error response (404, 500) into a Python exception instead of silently returning junk. Every HTTP call in this notebook uses it — fail loudly, not quietly.

**Router** — A node whose job is to decide *where to go next*. This notebook's supervisor is technically **not** a router: it extracts structure and then hands off along fixed edges. A true router would use a conditional edge to choose a branch at runtime.

## S

**Shared state** — One dictionary that every node reads from and writes into. It's the only communication channel between agents — there are no direct agent-to-agent messages here.

**START / END** — LangGraph's two built-in sentinel nodes. `add_edge(START, "supervisor")` marks the entry point; `add_edge("final_agent", END)` marks the exit.

**StateGraph** — The LangGraph builder class. You construct it with a state schema (`StateGraph(TravelState)`), add nodes and edges, then compile.

**Superstep** — LangGraph executes in rounds. In each round it runs every node whose prerequisites are satisfied, then merges all their state updates before starting the next round. This is why fan-out agents effectively run as a batch, and why a node reachable by two paths of different lengths can be scheduled more than once.

**Supervisor pattern** — A named multi-agent architecture where one lead agent reads the raw request, turns it into structured fields, and directs the specialists. Contrast with a **peer/debate pattern** (agents argue in a shared chat — see the AutoGen lecture) and a **pipeline** (a fixed A→B→C chain with no lead).

## T

**Tavily** — A search API built for LLMs: instead of raw HTML, it returns cleaned text snippets and, optionally, a synthesized `answer`. Used by both the hotel and cost agents.

**Temperature** — A dial from 0 upward controlling randomness. `temperature=0.2` is deliberately low: for extracting fields and reporting facts you want boring, repeatable output, not creativity.

**Tool** — Any capability the system can invoke that isn't the model itself — here, the five external APIs. Note the distinction from the ReAct lecture: there the *LLM* chooses which tool to call; here **the graph edges decide**, and the tools are called by plain Python.

**TypedDict** — A Python typing construct that describes the shape of a dictionary — which keys exist and what type each holds. `TravelState(TypedDict, total=False)` is the notebook's state schema. It's documentation and editor support only: Python does **not** enforce it at runtime.

**`total=False`** — The flag on `TypedDict` meaning "every key is optional". Essential here, because early in the run most keys (flights, hotels, cost…) genuinely don't exist yet.

[🔝 Back to top](#top)
