<a id="top"></a>
# Async Python for LLM Applications — Jargon Card

> **Use this file like a dictionary.** Skim it once (~7 min) before opening the notebook. Then keep it open in a side tab — when you hit an unknown word while reading, look it up here in 20 seconds instead of Googling for 5 minutes.
>
> **Companion:** read [`Async_Concurrency_Reading_Brief.md`](./Async_Concurrency_Reading_Brief.md) FIRST for the story; this card is just the dictionary.
>
> **Covers the notebook:** `Asynchronous_Programming_&_Concurrency_for_AI.ipynb` — sync-vs-async LLM calls, `async`/`await`, generators & streaming, `asyncio.gather`, task management (cancel/timeout/partial-failure), real LLM patterns, and a Matrix-Rain capstone (notebook + Gradio). Hands-on uses OpenAI's `AsyncOpenAI` client with `gpt-4o-mini`.

---

## A

**`async def` (coroutine function)** — The keyword that turns a normal function into a **coroutine function**. The twist beginners always trip on: *calling* it does **not** run the body — it hands you back a coroutine object that just sits there until you `await` it (or schedule it on the loop). In the notebook, `ask_async(prompt)` is defined with `async def`.

**`async for`** — A loop that consumes an **async generator**, `await`-ing the next item each iteration. The notebook uses `async for chunk in stream:` to pull LLM tokens as they arrive over the network — the core of streaming.

**`async with`** — A context manager that can `await` on enter/exit. Used with a `Semaphore` (`async with API_SEMAPHORE:`) to acquire a concurrency slot, parking the coroutine if all slots are taken.

**`asyncio`** — Python's standard-library framework for single-threaded concurrency built around an event loop, coroutines, and tasks. The whole notebook is an `asyncio` tour aimed at LLM I/O.

**`await`** — "Run this coroutine and give me its result — and while it's waiting on I/O, hand control back to the event loop so other coroutines can run." `await` is *both* how you get a coroutine's result *and* the yield point where cooperative multitasking happens. No `await` in a function body → nothing ever overlaps.

**`AsyncOpenAI` / `aclient`** — OpenAI's **asynchronous** SDK client. Every call (`await aclient.chat.completions.create(...)`) yields control while the HTTP round-trip is in flight. Its synchronous twin `OpenAI` blocks the thread — the notebook uses `OpenAI` once, only to make the "sync is slow" demo honest.

## B

**Blocking (call)** — An operation that holds the thread hostage until it finishes, so *nothing else can run* meanwhile. A synchronous HTTP request to an LLM blocks; three of them cost the **sum** of their latencies. The villain the whole notebook is fighting.

**Background task** — A coroutine scheduled with `create_task` that starts running immediately while your main coroutine keeps doing other work — "fire and remember." You `await` the task later to collect its result.

## C

**`CancelledError`** — The exception `asyncio` raises *inside* a coroutine when its task is cancelled (via `task.cancel()`). Catch it to clean up (close files, release locks), then **re-raise** so the cancellation is honored. The notebook's `long_llm_call()` shows the pattern.

**Concurrency** — Structuring a program so many tasks are *in progress* at once, making progress by interleaving — not necessarily running at the same instant. `asyncio` gives concurrency on **one** thread by switching at every `await`. (Contrast *parallelism*.)

**Cooperative multitasking** — The scheduling model `asyncio` uses: the event loop only switches coroutines at `await` points — each coroutine must *voluntarily* yield by awaiting. Nothing is preempted. The upside beginners love: **between `await`s, only one coroutine runs, so shared state needs no locks.**

**Coroutine (object)** — What you get when you *call* an `async def` function: a paused, not-yet-run unit of work. It runs only when awaited or scheduled. Calling without awaiting is the #1 bug (Python warns "coroutine was never awaited").

**`create_task` (`asyncio.create_task`)** — Schedules a coroutine to start running **now**, in the background, returning a `Task` handle you can `await`, `cancel`, or check `.done()` on. Use it (over `gather`) when you want work running while you do other things, or need a handle to cancel later.

## E

**Event loop** — The single-threaded scheduler at the heart of `asyncio`. It runs one coroutine until that coroutine hits an `await` on I/O, then pivots to another that's ready, and so on. Think **one chef juggling many pots**, stirring whichever is ready. In a notebook the loop is already running, which is why top-level `await` works.

## F

**Fan-out / fan-in** — The pattern of dispatching many coroutines at once (fan-out, usually via `gather`) then merging their results into one structure (fan-in, e.g. sorting into a ranked list). The notebook's LLM-as-judge scoring fans out 7 passages, fans in a novelty ranking.

**`FIRST_COMPLETED`** — A `return_when` mode for `asyncio.wait` that returns as soon as *any* coroutine finishes (the rest stay pending, ready to cancel). The basis of "return whichever LLM answers first" — an exercise in the notebook.

## G

**`gather` (`asyncio.gather`)** — The workhorse for parallel I/O: launches all the given coroutines **at once** and waits for **all** to finish. Two things to remember: results come back in **input order** (not completion order), and by default *one failure cancels the rest* — pass `return_exceptions=True` to change that. Turns "sum of latencies" into "max of latencies."

**Generator** — A function using `yield` that produces values **lazily**, one at a time, pausing between them. Regular generators are *synchronous* lazy sequences; the notebook fakes a tokenizer with one (`for word in sentence.split(): yield word`).

**Gradio** — A Python library for quick web UIs. Its event handlers can be **async generators** — each `yield` becomes a live UI update — which pairs perfectly with `asyncio`. The capstone wraps the Matrix-Rain streams in a Gradio app with Start/Stop sliders.

## H

**`html.escape`** — Converts `<`, `>`, `&` in a string to safe HTML entities. Essential when injecting **model output** into an HTML display (model text can contain HTML-looking fragments that would break the page). The Matrix-Rain renderer escapes every streamed chunk.

## I

**I/O-bound** — A workload whose time is dominated by *waiting* — for a network response, disk, or database — not by computation. LLM API calls are the textbook case, and exactly where async shines (you overlap the waiting). Contrast *CPU-bound*.

**CPU-bound** — A workload limited by raw computation (matrix math, parsing, crypto). Async does **not** help here — heavy CPU work in a coroutine *blocks the event loop*. Offload it with `asyncio.to_thread` or a process pool. (Named as async's "bad fit.")

## L

**Latency hiding** — The core payoff of async: while one request waits on the network, the event loop runs others, so their wait times **overlap** instead of stacking. Async makes no single call faster — it stops you from *idling* during the wait.

## P

**Parallelism** — Multiple things executing *literally simultaneously* (e.g. on multiple CPU cores). Distinct from *concurrency*: `asyncio` is concurrent but single-threaded, so it gives overlap of *waiting*, not simultaneous computation. For true CPU parallelism use multiprocessing.

**Partial failure** — When some coroutines in a batch fail and others succeed. Handled with `gather(..., return_exceptions=True)` (collect exceptions as values) and/or a per-task `try/except` that returns a structured error dict — so one bad LLM call never kills the batch.

## R

**`return_exceptions=True`** — The `gather` flag that makes it **collect** raised exceptions as items in the result list instead of re-raising and cancelling siblings. Lets the healthy calls finish. The notebook mixes valid and invalid model names to demonstrate it.

## S

**`asyncio.Semaphore`** — A counter that lets at most *N* coroutines into a section at once; the rest park on `async with sem:` until a slot frees. The idiomatic way to **cap concurrency** so a fan-out of 1,000 calls doesn't blow past API rate limits. The notebook caps in-flight LLM calls at 3.

**Streaming** — Emitting a response incrementally (token by token) instead of all at once. Implemented as an **async generator** of chunks (`stream=True` on the OpenAI call). Wins: perceived latency drops (first word in ~100 ms), you can pipeline downstream work, and cancellation is cheap.

**`asyncio.sleep`** — The **non-blocking** sleep: it yields to the event loop for the duration, so other coroutines run meanwhile. Its evil twin `time.sleep` blocks the *entire* loop — using it in async code is a classic pitfall.

## T

**Task (`asyncio.Task`)** — A coroutine that's been scheduled onto the loop (via `create_task`) and is now running/queued. Gives you a handle to `await` its result, `cancel()` it, or query `.done()`.

**`task.cancel()`** — Requests cancellation of a running task, which surfaces as `CancelledError` inside the coroutine at its next `await`. Pair with a `try/except asyncio.CancelledError` for cleanup. The Gradio Stop button cancels every in-flight stream this way.

**Top-level `await`** — Writing `await something()` directly in a notebook cell (no wrapper). Works because Jupyter/Colab already run an event loop. In a plain `.py` script you'd instead call `asyncio.run(main())`.

## W

**`asyncio.wait_for`** — Wraps an awaitable with a **timeout**: if it doesn't finish in time, it's cancelled and `TimeoutError` is raised. Your tool for enforcing a latency budget (`await asyncio.wait_for(call, timeout=0.3)`), then falling back to a cached answer or retry.

**`asyncio.wait`** — Lower-level than `gather`: awaits a set of tasks and returns `(done, pending)` sets, controllable with `return_when` (e.g. `FIRST_COMPLETED`). Use when you need first-to-finish or fine-grained control rather than "all results in order."

## Y

**`yield`** — Pauses a function and emits a value *without ending it*, resuming where it left off on the next request. Makes a function a **generator**; combined with `async def` it makes an **async generator**. LLM token streaming is "literally an async generator of tokens."

---

## The one-breath summary

- **LLM apps are I/O-bound**, so async is a big win: `async`/`await` + the **event loop** overlap the *waiting* on many network calls on a **single thread** (concurrency, not parallelism).
- **`gather`** runs coroutines in parallel (results in input order); **`create_task`** backgrounds one; **`wait_for`** bounds latency; **`cancel()`**/`CancelledError` end work early.
- **Resilience:** `return_exceptions=True` + per-task `try/except` so one bad call never kills the batch; **`Semaphore`** caps concurrency to respect rate limits.
- **Streaming = async generator of tokens** (`async def` + `yield`), which is also exactly what Gradio's streaming UIs consume.

[🔝 Back to top](#top)
