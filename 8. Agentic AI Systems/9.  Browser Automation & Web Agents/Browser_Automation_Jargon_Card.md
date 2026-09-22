<a id="top"></a>
# Browser Automation & Web Agents — Jargon Card

> **How to use.** Skim once (~5 min) before the notebook, then keep open as a side reference.
>
> **Companion:** read [`Browser_Automation_Reading_Brief.md`](./Browser_Automation_Reading_Brief.md) first.
>
> **Notebook:** `8_sep_Browser_automation_Classroom.ipynb` — 28 cells, Colab. Needs `GROQ_API_KEY` in Colab Secrets. **Restart the runtime after the install cell.**

---

## A

**Accessibility tree** — The structured, semantic view of a page that screen readers use: roles ("button", "textbox"), names, and states. It's the layer role-based locators query, and it's close to how an LLM is shown the page — which is why role locators and LLM agents pair so well.

**Actionability checks** — The five conditions Playwright verifies before it will click: the element is **attached** to the DOM, **visible**, **stable** (not animating), **receiving events** (not covered by an overlay), and **enabled**. Only when all five pass does the click fire. See *auto-wait*.

**Agent (browser-use)** — `Agent(task=..., llm=..., browser=..., use_vision=False)`. You give it a goal in English; it perceives the page, decides an action, acts, and repeats. `max_steps` caps the loop.

**Auto-wait** — Playwright's built-in retry: it re-checks the five actionability conditions until they pass or the timeout (default 30 s) expires. This is the single biggest reason Playwright suits LLM agents — **the model never has to reason about page-load timing.**

**Anti-bot heuristics** — Server-side detection of automation (odd user agents, impossible timing, the `navigator.webdriver` flag). The notebook's local-login script counters the last one with `args=["--disable-blink-features=AutomationControlled"]` and sets a realistic `user_agent`.

## B

**browser-use** — An open-source Python library layered on Playwright that gives an LLM a structured way to perceive and act on pages. The notebook's analogy: **Playwright is the steering wheel and pedals; browser-use is the chauffeur who can read road signs.** Four jobs: perception, a fixed action space, the perceive→think→act loop with retries, and extensibility.

**Browser → Context → Page** — The hierarchy to memorise. **Browser** = the whole Chrome process (expensive; start once). **Context** = one incognito window with its own cookies, storage and identity (cheap; this is how you get parallelism and multiple logins). **Page** = one tab inside a context.

## C

**Chromium / Firefox / WebKit** — The three browser engines Playwright drives. The notebook installs Chromium only.

## D

**`data-testid`** — An attribute developers add specifically for automation (`[data-testid='price']`). Stable, because changing it has no visual effect — nobody refactors it by accident.

**DOM (Document Object Model)** — The live tree of elements the browser builds from HTML and keeps mutating as JavaScript runs. It is *not* the same as the HTML you'd get from `requests.get()` on a modern site.

**DOM serialization with element indexing** — browser-use's default perception mode: walk the DOM, keep only visible interactive elements, and hand the LLM a numbered list (`[1]<button>Sign in</button>`, `[2]<input type="email"/>`). The model replies "click element 1". Fast, token-cheap, and it works with text-only models.

## H

**Headless** — Running the browser with no visible window. **Mandatory in Colab** (no display). Run `headless=False` locally when you need to watch, or log in by hand.

## I

**Infinite scroll** — Content that loads as you scroll. The pattern: scroll, wait, count items, stop when the count stops changing — **always with a hard iteration cap**. The notebook caps at 8 scrolls and gets 50 quotes.

## L

**Locator** — Playwright's handle on an element, e.g. `page.get_by_role("button", name="Add to cart")`. Locators are *lazy*: they re-resolve at use time, so they survive re-renders that would break a stored element reference. Stability ranking, best to worst: **role → text/label → `data-testid` → CSS class → XPath**.

## M

**`message_context`** — The browser-use 0.1.40 hook for injecting extra standing instructions into the agent — its policies, the SOP you hand the chauffeur. (Later versions add `extend_system_message`.)

## N

**`nest_asyncio`** — A shim that lets you start an event loop inside one that's already running. Needed because Colab runs its own loop and Playwright's async API wants one; `nest_asyncio.apply()` makes both coexist.

**`networkidle`** — A wait condition meaning "no network requests for 500 ms". **Overused and usually wrong** on modern sites: Flipkart, Amazon and friends never truly go idle because of background pings, so the wait times out. Prefer waiting on a specific element or URL.

## P

**Playwright** — Microsoft's browser-automation library. Drives a *real* browser, so JavaScript executes normally. Three features that matter for agents: auto-waiting, built-in network interception, and cheap parallelism via multiple contexts in one process.

**Pydantic** — A Python library for declaring data shapes as classes (`class HNStory(BaseModel): title: str; points: int`) and validating that real data matches. Used here as the final gate on LLM-extracted output.

## Q

**`quote_plus`** — URL-encodes a string so it can be dropped into a query parameter safely (spaces become `+`).

## R

**Role-based locator** — `get_by_role("button", name="Login")`. The most robust strategy, because it targets what the element *is* and what it's *called*, not where it sits or which CSS class it happens to carry this week.

## S

**Selector strategies** — See *Locator*. The rule for agents: prefer role and text, because they match how a human (and an LLM reading the accessibility tree) describes a page. **Avoid absolute XPath** — `/html/body/div[3]/...` breaks the moment a designer adds a wrapper div.

**SPA (Single Page Application)** — A site that ships a near-empty HTML shell and builds the page with JavaScript. `requests.get(url).text` returns almost nothing. Playwright runs a real browser, so the JS executes — but you must now decide *when* the page is "done".

**Strict JSON Schema (Groq)** — Groq's structured-output mode requires `additionalProperties: false` on **every** object in the schema, and all fields listed in `required`. Miss one and the request is rejected.

**`storage_state`** — Cookies plus `localStorage`, saved to a JSON file after a successful login and reloaded on later runs via `new_context(storage_state="file.json")`. The notebook's analogy: **the "Remember me on this device" checkbox.** ⚠️ Treat the file like an SSH key — anyone holding it can act as you. Never commit it.

**Structured output** — Forcing the model to emit JSON matching a declared schema, then validating it with Pydantic, instead of parsing prose. The reliability difference between a demo and a pipeline.

## U

**`use_vision`** — browser-use's perception switch. `False` = DOM-only mode (required for text-only models like `openai/gpt-oss-120b`). `True` = screenshot mode: browser-use draws numbered bounding boxes on a screenshot and sends it to a vision model — more reliable on visually complex pages (Canva, Figma), far more expensive in tokens.

**`user_agent`** — The string the browser sends identifying itself. Playwright's default announces automation; overriding it with a realistic Chrome string is a common first step against naive bot filters.

## W

**`wait_for_selector` / `wait_for_url`** — Explicit waits for a specific element or URL pattern. The preferred alternatives to `networkidle` when you need more control than auto-wait gives you.

[🔝 Back to top](#top)
