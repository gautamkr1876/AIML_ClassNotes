<a id="top"></a>
# Browser Automation & Web Agents — Reading Brief

> **Read once, end to end, before the notebook.** ~11 min.
>
> **Side reference:** [`Browser_Automation_Jargon_Card.md`](./Browser_Automation_Jargon_Card.md).
>
> **Notebook:** `8_sep_Browser_automation_Classroom.ipynb` — 28 cells, Colab. Setup takes 2–4 minutes and **you must restart the runtime afterwards**. One key: `GROQ_API_KEY` in Colab Secrets.

---

## 🎯 30-second TL;DR

**Playwright gives an agent eyes and hands. browser-use gives it a chauffeur that can read road signs.** That one sentence is the notebook's spine.

The first half is Playwright: how a real browser is structured (Browser → Context → Page), how to find elements robustly, why **auto-wait** means an LLM never has to reason about page-load timing, how to survive JavaScript-built pages and infinite scroll, and two ways to stay logged in. The second half hands the wheel to an LLM: `browser-use` serialises the page into a numbered list of clickable things (`[1]<button>Sign in</button>`), the model says "click element 1", and the loop repeats.

The pragmatic twist: after demoing the full autonomous agent, the notebook's two most polished examples (Hacker News, BBC) **go back to plain Playwright scraping plus one Groq structured-output call**. Deterministic scrape + LLM-for-structure is cheaper, faster and more reliable than autonomy whenever you already know the page.

---

## 🗺️ Agenda — what the notebook teaches, in order

| § | Cells | Topic | The one idea |
|--:|------|-------|--------------|
| — | 0–2 | Setup + Colab secrets | pin versions; restart the runtime |
| 1 | 3 | Why agents need a browser | most of the web isn't an API |
| 2 | 4–6 | Playwright basics | Browser → Context → Page |
| 3 | 7–9 | Locator strategies | role > text > testid > CSS > XPath |
| 4 | 11 | Auto-wait | five actionability checks, then click |
| 5 | 12–13 | SPAs + infinite scroll | wait on an element, always cap the loop |
| 6 | 14–16 | Auth pattern 1: `storage_state` | log in once, reuse the session file |
| 7 | 17–18 | Auth pattern 2: programmatic login | fill the form, then save the state |
| 8 | 20–21 | **browser-use deep dive** | DOM serialization vs vision mode |
| 9 | 23–24 | Structured output with Pydantic | scrape deterministically, structure with the LLM |
| 10 | 25–26 | Policies + BBC extraction | `message_context` is the agent's SOP |

---

## 🧠 The big idea — the chauffeur

You own a car. **Playwright is the steering wheel and pedals**: precise and mechanical — `page.click("button.add-to-cart")` clicks that button and nothing else. Perfect when you know the page.

The gap opens when you *don't*. You can't write `page.click(...)` for a site you've never seen, whose class names you don't know, which redesigns next month. **browser-use is the chauffeur who can read road signs.** You say "go to Hacker News, find the top story, tell me its title and points", and it looks at the page, identifies the interactive elements, picks one, clicks, looks again.

The key insight is *how* it looks. It doesn't hand the model raw HTML — it hands it a numbered menu:

```
[1]<button>Sign in</button>
[2]<input type="email" placeholder="Email"/>
[3]<a>Forgot password?</a>
```

The model's entire action vocabulary is "click element 1", "type into element 2", "scroll", "go back". That constraint is what makes it reliable — the model can't hallucinate a CSS selector when its only options are numbers on a list it was just shown.

And the corollary the notebook demonstrates by example: **once you know the page, fire the chauffeur and drive yourself.** Autonomy costs tokens, latency and determinism.

---

## 📖 Core concept primers

### 1. Browser → Context → Page

> **🪜 Mental model:** the Chrome process, an incognito window, a tab.

```
Browser   = the entire Chrome process   (expensive to start — start once)
Context   = one incognito window        (own cookies, storage, identity)
Page      = one tab inside a context
```

**Why the middle layer matters:** contexts are cheap and fully isolated, so five contexts in one browser process = five simultaneous logged-in identities scraping in parallel, with no cookie bleed. That's Playwright's parallelism story — and why `storage_state` and `user_agent` are set on the **context**, not the browser or the page.

### 2. Locators — find elements the way a human would

> **🪜 Mental model:** "the blue Login button" survives a redesign; "third div, second span" does not.

| Strategy | Example | Stability |
|---|---|---|
| Role-based | `get_by_role("button", name="Add to cart")` | ✅ best |
| Text / label | `get_by_text("Forgot password?")` | ✅ good |
| `data-testid` | `locator("[data-testid='price']")` | ✅ good |
| CSS class | `locator("button.css-1xyz9q")` | ⚠️ fragile |
| XPath | `locator("xpath=/html/body/div[3]/…")` | ❌ avoid |

CSS classes on modern sites are often build-generated hashes that change on every deploy; absolute XPath breaks when anyone adds a wrapper div.

**The agent-specific reason to prefer role and text:** they match how a human — and an LLM reading the accessibility tree — *describes* a page, so the model's reasoning vocabulary and your selectors' vocabulary become the same vocabulary.

### 3. Auto-wait — why Playwright fits LLM agents

> **🪜 Mental model:** a patient assistant who waits for the lift doors to actually open before stepping in.

Before any `click`, Playwright checks five things and retries until they pass or the timeout expires: attached to the DOM, visible, stable (not animating), receiving events (not covered by an overlay), enabled.

**Why this is huge for agents:** without it the model would have to reason about timing — "has the page loaded? should I wait?" — at every step. Instead it says "click element 7" and the framework handles reality.

For more control, wait on something *specific*:
```python
await page.wait_for_selector(".results", state="visible")   # ✅ preferred
await page.wait_for_url("**/dashboard")                     # ✅ good
await page.wait_for_load_state("networkidle")               # ⚠️ usually wrong
```
`networkidle` means "no network requests for 500 ms" — a condition busy commercial sites never reach, because of analytics pings and polling. It will time out.

### 4. SPAs and infinite scroll

Most modern consumer sites ship a near-empty HTML shell and build the page with JavaScript, so `requests.get(url).text` gets you almost nothing. Playwright runs a real browser and the JS executes — but now *you* must decide when the page is done. Three patterns, best first: **wait for the specific element you need**, **intercept the underlying API call** for clean JSON, or **scroll-and-count** for infinite feeds.

The scroll loop, from cell 13:
```python
for _ in range(8):                        # ← ALWAYS cap this
    await page.mouse.wheel(0, 5000)
    await page.wait_for_timeout(1200)
    current = await page.locator(".quote").count()
    if current == previous: break         # nothing new loaded → stop
    previous = current
```
Result: 50 quotes loaded. The cap is not optional — an uncapped scroll on a truly infinite feed runs until something times out.

### 5. Two authentication patterns

> **🪜 Mental model:** `storage_state` is the "remember me on this device" checkbox; programmatic login is typing your password every time.

**Pattern 1 — `storage_state` (the production one).** Logging in from scratch every run is slow, trips anti-bot heuristics, and may trigger 2FA. Instead: log in once by hand (locally, `headless=False`), save cookies + `localStorage` to JSON, and pass `new_context(storage_state="github_state.json")` on every later run — you start already logged in.

⚠️ **That file is a credential.** Anyone holding it can act as you. `.gitignore` it; encrypt it in production.

**Pattern 2 — programmatic login.** Fill the form in code. Right for rotating accounts, service accounts, or fresh onboarding. The demo uses a practice site with published test credentials and then — the nice touch — **saves `storage_state` afterwards**, combining both patterns.

### 6. How the LLM sees the page — DOM mode vs vision mode

**Mode 1 (default): DOM serialization with element indexing.** Walk the DOM, drop invisible elements, emit a numbered list of interactive ones. Fast, token-efficient, robust, and it works with **text-only models** — which is why the notebook sets `use_vision=False` for Groq's text-only `openai/gpt-oss-120b`.

**Mode 2: vision.** Screenshot the page, draw numbered bounding boxes, send the image to a vision model. More reliable on visually complex apps (Canva, Figma) where the DOM is uninformative; considerably more expensive.

---

## 🔥 The headline comparison — at a glance

The two ways this notebook gets data off a page:

| | Full agent (browser-use) | Scrape + structure (cells 24, 26) |
|---|---|---|
| Who picks the actions | the LLM, step by step | you, in Playwright code |
| Page knowledge needed | none | you must know the selectors |
| LLM calls | one per step (`max_steps=8`) | exactly one, at the end |
| Determinism | low — different path each run | high |
| Output shape | free text | Pydantic-validated JSON |
| Use it when | the page is unknown or changing | the page is known and stable |

The Hacker News example illustrates it cleanly: Playwright deterministically pulls `tr.athing` rows and their sibling subtext (`424 points by fnthawar2 3 hours ago | hide | 292 comments`), then **one** Groq call with a strict JSON schema turns that messy string into `{title, points, url, comments}`, validated by Pydantic. The LLM does the one thing it's uniquely good at — parsing unstructured text — and nothing else.

---

## 🧮 Shapes to memorise

**1. The hierarchy**
```
Browser → Context → Page
```
*In words:* one expensive process, holding cheap isolated windows, each holding tabs. Set identity (cookies, user agent) on the **context**.

**2. The minimal Playwright script**
```python
async with async_playwright() as p:
    browser = await p.chromium.launch(headless=True)
    context = await browser.new_context()
    page    = await context.new_page()
    await page.goto(url)
    ...
    await browser.close()
```
*In words:* start the engine, open a window, open a tab, go somewhere, do things, shut down. Every example in the notebook is this shape.

**3. The scroll-until-stable loop**
```
repeat up to N times: scroll → wait → count items → stop if count unchanged
```
*In words:* keep scrolling while new things keep appearing, but never more than N times. Worked example: 8 iterations, 1.2 s wait each, `.quote` count settles at 50.

---

## 🗺️ Notebook reading map

| Cells | What it teaches | How to read |
|---|---|---|
| 0–2 | Install, `nest_asyncio`, Colab secrets | **Read the pin comments** — they document real dependency failures |
| 3 | Why agents need browsers | **Skim** (image-only) |
| 4–6 | Playwright intro, hierarchy, hello-world | **Focus on the hierarchy**; the Wikipedia fetch is trivial |
| 7–9 | Locator table + Wikipedia search demo | **Focus on the table.** Note `fulltext=1` forces a deterministic results page |
| 11 | Auto-wait + explicit waits | **Focus.** The five checks and the `networkidle` warning |
| 12–13 | SPAs + infinite scroll | **Read normally.** Note the safety cap |
| 14–16 | `storage_state` | **Focus.** Cell 16 is *expected* to print "No session file" |
| 17–18 | Programmatic login | **Read normally.** Ends by saving state — both patterns combined |
| 19 | Session break marker | Skip |
| 20–21 | browser-use deep dive + live agent | **Focus on cell 20's prose** — DOM vs vision is the core concept |
| 23–24 | Pydantic + Hacker News structured extraction | **Focus.** The scrape/structure split is the practical takeaway |
| 25–26 | `message_context` + BBC extraction | **Read normally.** Note Groq's `additionalProperties: false` requirement |

---

## ⚠️ Gotchas

1. **Restart the runtime after the install cell.** The notebook says so in a comment. Skipping it gives you `module langchain has no attribute verbose` from a stale pre-installed LangChain.
2. **Version pins are load-bearing.** The comments record the actual failure: `playwright==1.47.0` + `pydantic==2.9.2` produced `ResolutionImpossible` and nothing installed. Don't "modernise" them casually.
3. **`headless=True` is mandatory in Colab** — there's no display. The `headless=False` script in cell 15 is explicitly for running on your own laptop.
4. **Cell 16 printing "No session file at github_state.json" is correct behaviour**, not a failure. You haven't uploaded one.
5. **`networkidle` is a trap.** Wait on a specific selector or URL instead.
6. **Never commit `storage_state` JSON.** It is a live session credential.
7. **Two different models appear.** Cell 21's browser-use agent uses `qwen/qwen3.6-27b`; the structured-output cells use `openai/gpt-oss-120b`. Not an error — just don't assume one model throughout.
8. **`max_tokens=512` and `max_steps=8` are rate-limit defences**, noted in the comments. Uncapped agent loops get you throttled fast.
9. **Groq strict JSON needs `additionalProperties: false` on every object** and every field in `required`. Miss one and the call is rejected.
10. **The BBC scraper takes every `<a>` with text ≥ 15 characters.** Crude on purpose — it's the LLM's job to sort headlines from navigation chrome downstream.

---

## ✅ Walk-away checklist

- [ ] The Browser → Context → Page hierarchy, and why context is the useful middle layer.
- [ ] The locator stability ranking, and why role/text locators suit LLM agents specifically.
- [ ] The five actionability checks, and why auto-wait matters more for agents than for humans.
- [ ] Why `requests.get()` fails on an SPA and Playwright doesn't.
- [ ] The two auth patterns, when each applies, and why the state file is a credential.
- [ ] DOM-serialization vs vision perception, and why this notebook uses DOM-only.
- [ ] When to use a full agent versus deterministic scrape + one structuring call.

---

## 🎯 5-question self-check

1. You need to scrape the same site as three different logged-in users, in parallel, in one script. Which layer of the hierarchy do you create three of, and why not the others?
2. A test clicking `button.css-1a2b3c` passes today and fails after a deploy, with the button visually unchanged. Diagnose it and give the fix.
3. Your agent must click a button that appears 2 seconds after page load, behind a fade-in animation. How much waiting code do you write, and why?
4. You're extracting the top 5 Hacker News stories every hour. Would you use a browser-use agent or Playwright + a structuring LLM call? Give two reasons.
5. A colleague commits `auth_state.json` so teammates can run the scraper. What's wrong, and what should they do instead?

<details>
<summary><strong>Answers</strong></summary>

1. **Three contexts, inside one browser.** Contexts are cheap and fully isolated — separate cookies, storage and identity — so three logged-in sessions can't bleed into each other. Three *browsers* would work but wastes a whole Chrome process each. Three *pages* in one context would share cookies, so all three tabs would be the same user.

2. The class is a build-generated hash that changed on deploy — a fragile CSS selector, second-from-bottom of the stability ranking. Fix by targeting what the element *is* and is *called*: `page.get_by_role("button", name="Add to cart")`. If the team controls the code, adding a `data-testid` is the other durable answer.

3. **None.** Playwright's auto-wait already retries the five actionability checks — attached, visible, stable, receiving events, enabled — until they pass or the 30-second timeout expires. "Stable (not animating)" covers the fade-in specifically. Adding a manual `sleep` makes the script slower and *less* reliable. Only reach for `wait_for_selector`/`wait_for_url` when you need to wait on something other than the element you're about to act on.

4. **Playwright + one structuring call.** (a) The page is known and stable — you can write `tr.athing` once and it keeps working, so you're paying an LLM every step for a decision that never changes. (b) Determinism and cost: one LLM call per run instead of one per agent step, and the same result every time, which matters for something running hourly. An agent earns its cost only when the page is unknown or changing.

5. `auth_state.json` contains live session cookies — **anyone with the file can act as that account**, no password needed. It's an SSH key, not config. It should be `.gitignore`d, and if already committed, rotated (log out everywhere to invalidate the sessions) and purged from history. Teammates should each run the one-time local login to generate their own; in production the file belongs in a secret manager.

</details>

[🔝 Back to top](#top)
