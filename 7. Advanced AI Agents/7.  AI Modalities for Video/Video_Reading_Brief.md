<a id="top"></a>
# Video (AI Modalities) — Reading Brief

> **Read this ONCE, end to end, before opening the notebook.** Target time: ~22 minutes. By the time you reach the notebook, every word in it will already make sense — you'll be confirming what you know, not learning blind.
>
> **Side reference:** keep [`Video_Jargon_Card.md`](./Video_Jargon_Card.md) open in another tab while reading. When an unknown word appears, look it up there.
> **The notebooks:** `ai_modality_video.ipynb` (the main video pipeline) and `open_cv_demo.ipynb` (OpenCV image basics) in this folder.

---

## 🎯 30-second TL;DR

**You cannot hand a whole video to an LLM. You *convert* it into things the LLM already understands — a handful of timestamped frames plus a timestamped transcript — and reason over those.**

The notebook builds this end to end on a 57-second movie clip (`bribe.mp4`, from *The Naked Gun 2½*):

- The raw clip is **1367 frames** at 1920×1080 — **8.5 GB** if fully decoded. What actually gets sent to the model is **8 small frames ≈ 3.5 MB — 2403× smaller.**
- Consecutive frames are near-identical (**mean frame-to-frame change ≈ 3.52** on a 0–255 scale), which is *why* sampling only 8 frames loses almost nothing.
- **Frames alone are timeless** — the notebook labels each with its timestamp (`[frame @ 8.2s]`) so the model can reason *over time*.
- Add the **audio**: ffmpeg strips the sound track, Whisper transcribes it into **49 timestamped segments**, and fusing transcript + keyframes lets the model produce a timestamp-cited summary and even name the movie.
- The endgame: turn captions and transcript segments into **searchable `{timestamp, text}` chunks**, embed them, and you have **video RAG** — ask a question, get the exact moments back.

The one sentence to carry out: *"Do not send the entire video blindly — convert it into sampled frames + timestamps + transcript + searchable text chunks."*

---

## 🗺️ Agenda — what the notebook teaches, in order

1. **How to send an image to a vision LLM** — two API shapes: Chat Completions (`image_url`) and the newer Responses API (`input_image`, with token `usage`).
2. **Inspect the video (cheap)** — read fps, frame count, width, height *without* decoding; compute duration and the scary raw-decoded size (8.5 GB).
3. **Sample representative frames** — pick 8 evenly-spaced frames with `np.linspace`, shrink each, convert BGR→RGB, **keep the timestamps**.
4. **Prove temporal redundancy** — decode densely at tiny 64×64, measure mean-absolute-difference between frames; flat = redundant (sample freely), spikes = shot cuts.
5. **See the cost trade-off** — same frame at 480/256/128/64 px shrinks the bytes dramatically while staying legible; sampling + resizing *compound*.
6. **Encode + send ordered frames** — JPEG→Base64 data URLs, each tagged with its timestamp, all in one request, and ask the model to describe what happens across the clip.
7. **Extract + transcribe audio** — ffmpeg (`-vn`) → 16 kHz mono WAV → Whisper with `verbose_json` → 49 timestamped `{start, end, text}` segments.
8. **Fuse audio + video** — feed the timestamped transcript *plus* keyframes together; the model returns a 3-bullet, timestamp-cited summary and identifies the source movie.
9. **(OpenCV demo notebook)** — image fundamentals the pipeline rests on: BGR vs RGB, grayscale, resize/crop/flip/rotate, imread/imwrite.
10. **(Outline) build searchable chunks → embed → video RAG** — the conceptual destination: frame captions + transcript segments become embedded `{timestamp, text}` chunks you can semantically search.

---

## 🧠 The big idea — "video = images + a clock"

A vision LLM already knows how to look at **one picture**. It does **not** know what a video is, and it has **no sense of time**. So the whole trick of this notebook is a translation:

> **A video is just a fast flip-book of still images. Give the model a few of those images, and — crucially — stamp each one with *when* it happened. Images are the "what"; the timestamp is the "clock."**

Everything in the pipeline is a consequence of that one analogy:

- **Why sample instead of send everything?** A flip-book has hundreds of pages that barely differ from the one before. You don't need every page to follow the story — a few evenly-spaced pages tell it. (The notebook *measures* this: neighbouring frames differ by only ~3.52 out of 255.)
- **Why label timestamps?** If you hand someone shuffled, unlabelled flip-book pages, they can't tell you what happened *first*. The model is the same — it can't infer order or time from pixels, so you write it on each frame.
- **Why add audio?** The pictures show a tense conversation, but the *words* ("Lieutenant Drebin… bribe…") carry the plot. Combine both and the model understands the scene far better than either alone — that's **multimodal fusion**.
- **Why chunk + embed at the end?** Once every moment is a `{timestamp, text}` record, you can *search the video by meaning* — the foundation of **video RAG**.

Carry the flip-book-with-a-clock image into any interview and you can re-derive the entire pipeline.

---

## 📖 Core concept primers

Six primers cover the heart of the notebook. Each has a mental model, plain-English meaning, and the notebook's real numbers.

### 1. Frame sampling (and why it's safe)

> **🪜 Mental model:** you don't read every page of a flip-book to get the story — a few evenly-spaced pages are enough.

A **frame** is one still image; a video is a fast sequence of them. **Frame sampling** means picking a small, evenly-spaced subset instead of all of them. The notebook's clip has **1367 frames**; it keeps just **8**, chosen with `np.linspace(0, 1366, 8)` so they land at t = 0.0, 8.2, 16.4 … 57.4 s — spread evenly across the whole 57 seconds.

**Why it's safe:** consecutive frames are near-duplicates. Sending all 1367 would cost ~170× more tokens for almost no extra information. Uniform sampling gives broad **temporal coverage** (beginning, middle, end) at a tiny fraction of the cost — the difference between an 8.5 GB payload and a 3.5 MB one (the notebook prints "2403x smaller").

### 2. Temporal redundancy + the change signal (MAD)

> **🪜 Mental model:** hold your thumb over a movie and flick through it — most of the time nothing moves; occasionally the whole picture jumps (a cut).

**Temporal redundancy** is the fact that neighbouring frames look almost identical. The notebook *proves* it cheaply: decode the whole video at a tiny **64×64** grayscale (detail doesn't matter — you only need a change signal), then compute the **Mean Absolute Difference (MAD)** between consecutive frames — the average per-pixel change.

- **Low MAD** → frames barely changed → sampling is safe (dense frames would waste tokens).
- **A spike in MAD** → a big change, usually a **shot cut** (scene change) — a natural boundary for chunking long video later.

The notebook's clip averages **MAD ≈ 3.52** on a 0–255 scale — very low, confirming heavy redundancy. It's computed with `np.abs(np.diff(small, axis=0)).mean(axis=(1,2))`, where `np.diff` subtracts each frame from the next.

### 3. Timestamps — giving the model a clock

> **🪜 Mental model:** a photo has no date stamp; if you want the model to reason about *when*, you write the time on the back.

A vision model sees pixels, not a clock. So the notebook **preserves and attaches a timestamp to every frame** — literally inserting a text label `[frame @ 8.2s]` right before each image in the request. The comment in the code says it plainly: *"the model cannot infer time on its own — we have to tell it."*

Timestamps do double duty: (1) they let the model reason about *order and time* ("the argument escalates over the clip"), and (2) later, they're the **metadata** that lets RAG answer *"at what moment did X happen?"* by pointing you back to the exact second.

**Why it matters here:** with timestamps, the model's answer isn't just "two men argue" — it's a *timeline* ("0–5s: interrogation begins … 46–50s: a bribe is offered").

### 4. Cost & tokens — sampling and resizing compound

> **🪜 Mental model:** you're paying by the pixel — every frame you drop and every pixel you shrink is money saved.

LLMs bill in **tokens**; images become **visual tokens**, and *more pixels or higher `detail` = more tokens = more cost*. The notebook attacks cost on three fronts that **multiply together**:

1. **Fewer frames** (8 instead of 1367).
2. **Fewer pixels per frame** (resize the long side to ~512, then the notebook shows 480→256→128→64 px still stays legible while KB plummets).
3. **`detail="low"`** on each image, so the API reads a small fixed tile rather than full resolution.

The headline arithmetic: raw decoded = `1367 × 1080 × 1920 × 3 ≈ 8503.8 MB`; what you send = `8 × 288 × 512 × 3 ≈ 3.54 MB` → **2403× smaller**. Each frame ends up 4–32 KB as a Base64 JPEG. The trade-off to state in an interview: **more frames → better temporal coverage; higher resolution → better visual detail; both cost more.**

### 5. Audio → transcript (ASR with timestamps)

> **🪜 Mental model:** the pictures are the silent movie; the transcript is the subtitle track that carries the plot.

Frames show *what's on screen*; speech carries meaning frames can't. The notebook extracts the audio with **ffmpeg** (`-vn` drops the video, `-ac 1 -ar 16000` forces **mono, 16 kHz** — the compact format speech models expect) into a WAV, then runs **ASR (Automatic Speech Recognition)** via **Whisper** (`whisper-1`).

The key flag is `response_format="verbose_json"` with `timestamp_granularities=["segment"]`, which makes Whisper return **timestamped segments** — each a `{start, end, text}` record. The clip yielded **49 segments** (e.g., `[0.0-1.0s] Lieutenant Drebin, police squad.`); those timestamps let audio line up with the frames. Without audio the model sees a physical argument; with the transcript it learns the *content* — an interrogation about a bribe.

### 6. Multimodal fusion + the road to video RAG

> **🪜 Mental model:** hand the model both the subtitle track and a few screenshots at once — now it can connect what was *said* to what was *shown*.

**Fusion** means putting the timestamped transcript **and** a few keyframes into a single request so the model reasons over both. The notebook's fused prompt ("Summarize the clip in 3 bullets, each citing a timestamp") produces a genuinely grounded answer — and a follow-up ("What movie is this from?") correctly returns *The Naked Gun 2½ (1991)* from the dialogue cues.

The outline then points to the **destination**: turn every frame-caption (visual route) and transcript segment (audio route) into a `{timestamp, text}` **chunk**, **embed** each into a vector with `text-embedding-3-small`, keeping the timestamp as metadata. At query time you embed the question, compare it against all chunk vectors (**semantic search**), and return the closest moments — optionally feeding them to an LLM. That's **video RAG (Retrieval-Augmented Generation)**: search a video by meaning and get back the exact seconds.

---

## 🔥 The headline pipeline — at a glance

Real numbers from `ai_modality_video.ipynb` running on `bribe.mp4` (57.4 s, 1920×1080, 23.81 fps):

| Stage | What happens | Real numbers |
|---|---|---|
| **Inspect** | Read metadata, no decode | 1367 frames · 1920×1080 · 23.81 fps · 57.4 s · **8.5 GB** if decoded |
| **Sample** | 8 evenly-spaced frames, resized, BGR→RGB | 8 frames · resized to 288×512 · t = 0.0…57.4 s |
| **Redundancy** | Dense 64×64 decode → MAD | mean frame-to-frame change **≈ 3.52 / 255** (very low) |
| **Cost** | Sampling + resizing compound | send **3.54 MB** vs 8.5 GB raw → **2403× smaller**; frames 4–32 KB each |
| **Send frames** | JPEG→Base64, timestamp-labelled, `detail="low"` | model: *"Two men… tense conversation… escalates into a physical confrontation"* |
| **Audio** | ffmpeg → 16 kHz mono WAV → Whisper | **49** timestamped `{start,end,text}` segments |
| **Fuse** | Transcript + keyframes in one request | 3 timestamp-cited bullets; correctly names *The Naked Gun 2½ (1991)* |
| **RAG (outline)** | Chunk → embed → semantic search | `{timestamp, text}` chunks embedded with `text-embedding-3-small` |

**Central idea (from the outline, verbatim):** *"Do not send the entire video blindly. Convert it into a compact representation made of: sampled frames + timestamps + transcript + searchable text chunks."*

---

## 🧮 Key rules & formulas to memorise

This notebook is **mostly practical** (APIs, sampling, cost) — there are only a few small formulas, plus a set of rules of thumb. Memorise these:

### Formula 1 — raw decoded video size

```
bytes = n_frames × H × W × C        (C = 3 for colour, dtype uint8 = 1 byte)
```

**In words:** raw memory = number of frames × height × width × channels. **Worked example:** `1367 × 1080 × 1920 × 3 = 8,503,833,600 bytes ≈ 8.5 GB`. This is why you never keep a whole decoded video in memory — and why the file on disk (a few MB) is so much smaller (that gap is the **codec/compression**).

### Formula 2 — duration from metadata

```
duration = n_frames / fps
```

**In words:** total seconds = frame count divided by frames-per-second. **Example:** `1367 / 23.81 ≈ 57.4 s`.

### Formula 3 — evenly-spaced sample indices

```
indices = np.linspace(0, n_frames − 1, num_frames)   # then round to int
```

**In words:** ask NumPy for `num_frames` numbers spread evenly from the first frame (0) to the last (`n_frames − 1`). **Example:** `np.linspace(0, 1366, 8)` → 8 indices spanning the whole clip → timestamps 0.0, 8.2, 16.4 … 57.4 s.

### Formula 4 — the change signal (MAD)

```
mad[i] = mean( |frame[i+1] − frame[i]| )
```

**In words:** for each pair of neighbouring frames, take the absolute pixel differences and average them. **Low = redundant (sample freely); a spike = a shot cut.** The clip's average is ≈ 3.52 on a 0–255 scale.

### Rules of thumb (no formula, but always true here)

- **OpenCV is BGR** — convert with `cv2.cvtColor(..., COLOR_BGR2RGB)` before showing or sending, or colours swap.
- **Frames have no clock** — always attach a timestamp label to each frame you send.
- **Cost knobs multiply** — fewer frames × fewer pixels × `detail="low"` compound into huge savings.
- **JPEG (not PNG) for frames** — ~10× smaller for photos; `quality=80` is invisible to a vision model.
- **Whisper needs `verbose_json` + `segment` granularity** to return timestamps.

---

## 🗺️ Notebook reading map — where to spend your attention

**`ai_modality_video.ipynb`** (the main notebook):

| Cells | What it teaches | How to read |
|---|---|---|
| **1–8** | Setup, model choices (`gpt-5-nano`, Whisper, embeddings), two ways to send an image to a vision LLM | **Skim** — ~5 min. Note the two API shapes (Chat vs Responses). |
| **9–14** | Read video metadata cheaply; the 8.5 GB raw-size shock; fps trivia (24 vs 120) | **Read** — ~4 min. Absorb the size arithmetic. |
| **15–17** | `extract_frames` (sampling + resize + BGR→RGB + timestamps); the cost-vs-pixels comparison | **FOCUS** — ~8 min. This helper is the pipeline's spine. |
| **18–20** | Temporal redundancy: 64×64 dense decode, MAD change signal, shot-cut spikes | **Read carefully** — ~6 min. This is the "why sampling works" proof. |
| **21–23** | Build the ordered, timestamp-labelled frame request; model describes the clip | **FOCUS** — ~5 min. The core send. |
| **24–27** | ffmpeg audio extract → Whisper → 49 timestamped segments | **Read** — ~5 min. |
| **28–32** | Fusion: transcript + keyframes → timestamp-cited summary; identify the movie | **FOCUS — the payoff** — ~6 min. |

**`open_cv_demo.ipynb`** (supporting image basics):

| Cells | What it teaches | How to read |
|---|---|---|
| **1–7** | BGR vs RGB, imread/imwrite, grayscale, shapes | **Skim** — ~4 min. Foundational, not new if you've done CV. |
| **8–9** | Resize (interpolation choices), crop (NumPy slicing), flip, rotate | **Reference** — ~4 min. |

**Total notebook read time:** ~45 min for both. Add this brief's ~22 min → ~67 min, comfortably under an unassisted 90–120+ min blind read.

---

## ✅ Walk-away checklist

After the notebook, you should be able to say in your own words:

- [ ] **Why you can't send a whole video to an LLM** — it's timeless, image-only, and the raw decode is gigabytes (8.5 GB here); you send sampled frames + timestamps instead.
- [ ] **Why sampling 8 of 1367 frames loses almost nothing** — temporal redundancy; neighbouring frames barely differ (MAD ≈ 3.52).
- [ ] **How you give the model a sense of time** — attach a timestamp label to each frame; the model can't infer it.
- [ ] **The three cost knobs and that they compound** — fewer frames, fewer pixels, `detail="low"` → 2403× smaller here.
- [ ] **How audio joins the picture** — ffmpeg strips it, Whisper transcribes it into timestamped segments, fusion feeds transcript + keyframes together.
- [ ] **What "video RAG" means** — chunk every moment into `{timestamp, text}`, embed, semantically search, retrieve the matching seconds.
- [ ] **The OpenCV BGR gotcha** — convert BGR→RGB before displaying or sending.

If any feel shaky, return to the matching primer above.

---

## 🎯 5-question self-check

Answer in your head, then check below. **No peeking.**

1. A clip has **1367 frames** at **23.81 fps**. What's its duration, and roughly how many MB is one raw decoded **1080×1920×3** frame?
2. Why is it *safe* to send only 8 frames instead of all 1367 — and how does the notebook *measure* that it's safe?
3. You send 8 frames but the model's summary confuses the order of events. What did you most likely forget to include, and why does it matter?
4. Name the three cost knobs the notebook uses and state whether their savings add or multiply.
5. In one sentence: what is "video RAG," and what role do timestamps play in it?

---

<details>
<summary><b>Click to reveal answers</b></summary>

1. **Duration = 1367 / 23.81 ≈ 57.4 s.** One raw frame = `1080 × 1920 × 3 = 6,220,800 bytes ≈ 6.2 MB`. (All 1367 frames ≈ 8.5 GB — the notebook's headline number.)
2. **Safe because of temporal redundancy** — consecutive frames are near-identical, so a few evenly-spaced ones cover the story. The notebook **measures** it by decoding densely at 64×64 grayscale and computing the **Mean Absolute Difference (MAD)** between consecutive frames; it comes out ≈ 3.52 on a 0–255 scale (very low), and spikes would mark shot cuts.
3. **You most likely forgot the per-frame timestamp labels** (`[frame @ 8.2s]`). A vision model can't infer time or order from pixels — you have to write the timestamp on each frame so it can reason about what happens *over time*, not just within one frame.
4. **Fewer frames** (8 vs 1367), **fewer pixels per frame** (resize the long side to ~512 / down to 64px in the demo), and **`detail="low"`** on each image. Their savings **multiply (compound)** — the notebook reports the combined effect as 2403× smaller (3.54 MB vs 8.5 GB).
5. **Video RAG = turning every video moment into an embedded `{timestamp, text}` chunk so you can semantically search the video and retrieve the exact matching seconds (optionally feeding them to an LLM to answer).** Timestamps are the metadata that let a retrieved chunk point back to *when* in the video the answer lives.

</details>

---

[🔝 Back to top](#top) · [→ Jargon Card](./Video_Jargon_Card.md)
