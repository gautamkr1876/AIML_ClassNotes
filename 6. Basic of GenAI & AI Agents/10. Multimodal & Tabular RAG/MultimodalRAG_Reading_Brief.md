<a id="top"></a>
# Multimodal RAG — Reading Brief

> **Read this ONCE, end to end, before opening the notebook.** Target time: ~22 minutes. By the time you reach the notebook, every word in it will already make sense — you'll be confirming what you already know, not learning blind.
>
> **Side reference:** keep [`MultimodalRAG_Jargon_Card.md`](./MultimodalRAG_Jargon_Card.md) open in another tab while reading. When an unknown word appears, look it up there.
> **The notebook:** `L10 (Jun 16th)_ Multimodal RAG.ipynb` (~85 cells).
> **Prereqs:** the text-only RAG pipeline (`../8. ...RAG Introduction/`) and embeddings (`../9. Embeddings Deep Dive/`).

---

## 🎯 30-second TL;DR

Text-only RAG can't answer "find me the X-ray that looks like this" or "what does this IKEA diagram say." **Multimodal RAG** fixes that by retrieving and reasoning over **images**, not just text. The notebook builds *two* complete multimodal RAG systems:

> **System A:** CLIP (embed text + images into one space) → ChromaDB → **GPT-4o Vision** answers
> **System B:** ColPali (treat PDF pages as images, no OCR) → **Qwen2-VL** answers

The one idea that unlocks everything: **a shared embedding space where text and images become comparable vectors.** Once a sentence and a picture live at the same kind of coordinate, "search images with words" is just nearest-neighbor lookup — the exact same retrieval machinery from the text-RAG lecture, with a multimodal embedder swapped in.

---

## 🗺️ Agenda — what the notebook teaches, in order

1. **Why multimodal RAG?** — the scenarios text-only RAG can't handle (search by image, diagrams, screenshots).
2. **CLIP intuition** — dual encoder, shared embedding space, cross-modal search.
3. **How CLIP is trained** — contrastive learning, the symmetric loss, temperature (a from-scratch mini-CLIP walkthrough).
4. **System A setup** — load CLIP `ViT-B/32`, build a landmark knowledge base (text + image per item).
5. **Embed & store** — CLIP-embed text and images, visualize the text-image similarity matrix, load into ChromaDB.
6. **Multimodal search** — query by text *or* by image; inspect top-k results.
7. **Generation with GPT-4o Vision** — pack retrieved text + base64 images into one prompt; get a grounded answer.
8. **System B** — ColPali document retrieval over IKEA PDFs (pages as images) + Qwen2-VL visual QA, no OCR.

---

## 🧠 The big idea — one space for words and pixels

Text RAG works because text embeddings put similar *sentences* near each other. Multimodal RAG needs something stronger: a space where a *sentence* and an *image* of the same thing land near each other. That's exactly what **CLIP** provides.

**The transferable analogy: a bilingual dictionary with one shared "meaning" page.** Imagine English and French words that don't map word-to-word, but both point at the same picture of the concept. CLIP is that shared concept page for *text and images*: "a photo of the Eiffel Tower" (text) and an actual Eiffel Tower photo (pixels) both get encoded to nearly the same 512-dimensional vector. Once they share coordinates, every cross-modal operation becomes trivial geometry:

- **Text → Image:** embed the words, return the nearest image vectors. ("Show me beautiful natural formations carved by water" → Grand Canyon photo.)
- **Image → Image:** embed a query photo, return similar photos.
- **Image → Text:** embed a photo, return the nearest captions/documents.

The rest of the notebook is the familiar RAG loop — **Retrieve → Augment → Generate** — with two swaps: the *retriever* is now multimodal (CLIP or ColPali), and the *generator* is now a **vision-language model** (GPT-4o Vision or Qwen2-VL) that can actually *look at* the retrieved images while it writes the answer.

---

## 📖 Core concept primers

Five primers cover the heart of the notebook. Each has a **mental model**, plain-English meaning, a concrete detail from the notebook, and why it matters here.

### 1. CLIP & the shared embedding space

> **🪜 Mental model:** two cameras pointed at the same scoreboard — the text encoder and image encoder report different inputs but land on the same coordinate for the same concept.

**CLIP (Contrastive Language-Image Pre-training)** is a **dual encoder**: a text encoder and an image encoder trained *together* so their outputs live in one shared 512-dim space. OpenAI trained it on ~400 million internet image-caption pairs. The notebook loads `ViT-B/32` and embeds both modalities with `model.encode_text(...)` and `model.encode_image(...)`, L2-normalizing each so cosine similarity = dot product. **Why it matters here:** this single space is what lets a *text* query retrieve *images* — you embed the query (either modality) and do nearest-neighbor search against stored image vectors. The notebook's similarity-matrix heatmap visualizes it: the diagonal (each caption vs its own image) should score highest.

### 2. Contrastive learning (how CLIP learned this)

> **🪜 Mental model:** a matching quiz — given N photos and N captions shuffled, learn to draw the right lines between them.

Rather than predicting exact captions (very hard), CLIP solves a **matching** problem: for a batch of N image-text pairs, build an N×N similarity matrix and make the **diagonal** (true pairs) large while off-diagonal (mismatches) stay small. The loss is **symmetric cross-entropy** — computed image→text *and* text→image, then averaged: `(loss_i2t + loss_t2i)/2` — and a learnable **temperature** (init 0.07) sharpens the distribution. **Why it matters here:** the notebook includes a from-scratch mini-CLIP so you *see* the similarity matrix and loss; understanding "pull matching pairs together, push the rest apart" demystifies why CLIP embeddings are cross-modally comparable at all. (It's the same contrastive idea behind triplet loss from the Embeddings lecture, generalized to two modalities.)

### 3. Multimodal retrieval with ChromaDB

> **🪜 Mental model:** the same filing cabinet from text RAG — you just also file photographs, indexed by the same kind of vector.

Once CLIP turns everything into 512-dim vectors, storage and search are *identical* to text RAG. The notebook computes embeddings for each landmark's text and image, then stores them in **ChromaDB** collections configured for cosine space (`metadata={"hnsw:space": "cosine"}`). A query (text or image) is CLIP-embedded and run through `collection.query(query_embeddings=..., n_results=top_k)`. **Why it matters here:** it shows multimodal RAG reuses *all* the vector-DB machinery you already learned — nothing new in storage/search, only the embedder changed.

### 4. Generation with a Vision-Language Model

> **🪜 Mental model:** handing a colleague the relevant photos *and* notes, then asking the question — they can point at what's in the picture.

Retrieval finds relevant images; a **VLM** turns them into an answer. The notebook builds a multimodal prompt: retrieved **text** context as a string, plus retrieved **images** converted to **base64** and attached as image parts, plus the user's question — all sent to **GPT-4o Vision** (`gpt-4o`). The model reasons over text *and* pixels together. **Why it matters here:** this is the "G" in multimodal RAG. A text-only LLM literally cannot see the retrieved images; a VLM can describe what's *in* them ("the diagram shows the legs attaching with 4 screws"). The notebook resizes images before base64 to save tokens.

### 5. Document RAG without OCR (ColPali + Qwen2-VL)

> **🪜 Mental model:** instead of transcribing a scanned manual into text and hoping the transcription is right, just *show the page* to a model that can read it.

System B tackles real documents (IKEA assembly PDFs full of diagrams and tables). The classic approach — OCR the page into text, then do text RAG — breaks on visual layouts. Instead: **ColPali** (`vidore/colpali-v1.2`, via the **byaldi** wrapper) indexes each **PDF page as an image** and does semantic search over those page-images; the top pages are handed *as images* to **Qwen2-VL** for visual question answering. `pdf2image` converts pages to images first. **Why it matters here:** it's the notebook's headline practical win — *no OCR pipeline*, yet it answers "How do I attach the legs to the table?" by retrieving and reading the right diagram page. Tradeoff: needs a big GPU (≥24 GB VRAM).

---

## 🔥 Two systems — at a glance

| | **System A: CLIP + GPT-4o Vision** | **System B: ColPali + Qwen2-VL** |
|---|---|---|
| **Knowledge base** | Landmarks (text + 1 photo each) | IKEA assembly PDFs (MALM, BILLY, …) |
| **Retriever** | CLIP `ViT-B/32` (512-dim) | ColPali (`vidore/colpali-v1.2`) |
| **What's embedded** | Text *and* images, shared space | PDF **pages as images** (no OCR) |
| **Vector store** | ChromaDB (cosine) | byaldi's built-in index |
| **Generator** | GPT-4o Vision (API) | Qwen2-VL-7B-Instruct (local GPU) |
| **Query shown** | "natural formations carved by water" → Grand Canyon | "How do I attach the legs to the table?" |
| **Hardware** | CPU-friendly (CLIP small) + API | GPU ≥24 GB VRAM |

Both follow **Retrieve → Augment → Generate**; they differ only in *how* images enter the pipeline (CLIP embeds photos; ColPali embeds page-images) and *which* VLM generates.

---

## 🧮 The one formula to internalize — CLIP's symmetric loss

```
labels = [0, 1, 2, ..., N-1]          # for image i, the matching text is also at index i
loss = ( cross_entropy(logits_per_image, labels)
       + cross_entropy(logits_per_text,  labels) ) / 2
```

**Word-by-word translation:** "Build an N×N similarity matrix for a batch of N image-text pairs. The correct match for row *i* is column *i* (the diagonal). Score how well images pick their true caption (image→text) *and* how well captions pick their true image (text→image), then average the two." Minimizing this drives matching pairs' similarity up and mismatched pairs' down — which is *exactly* what creates the shared space.

**Worked reading:** for a batch of 4 pairs, a perfect model puts all probability mass on the diagonal of the 4×4 matrix; cross-entropy against `labels=[0,1,2,3]` is then ~0. Any off-diagonal mass (image matched to the wrong caption) raises the loss.

---

## 🗺️ Notebook reading map

| Cells | What it teaches | How to read |
|---|---|---|
| 0–1 | Why multimodal RAG; use cases | **Focus.** Motivation. |
| 2–7 | CLIP intuition: dual encoder, shared space, cross-modal table | **Focus.** Core concept. |
| 8–19 | From-scratch mini-CLIP: model, contrastive loss, temperature | **Read.** See the similarity matrix + loss; don't memorize code. |
| 20–24 | System A setup: install, load `ViT-B/32`, landmark knowledge base | **Skim.** Note 512-dim, the KB shape. |
| 25–32 | Embed text+images, similarity-matrix heatmap, load ChromaDB | **Focus.** Watch the diagonal in the heatmap. |
| 33–36 | `MultimodalSearchEngine`: text query, image query | **Focus.** Cross-modal search in action. |
| 37–45 | GPT-4o Vision: base64 images, build multimodal prompt, `MultimodalRAG` | **Focus.** The generation step. |
| 46–52 | System B intro: ColPali + Qwen2-VL, install, download IKEA PDFs | **Read.** Note "no OCR," GPU requirement. |
| 53–66 | PDF→images, load+index ColPali, test retrieval, score plots | **Focus.** Page-as-image retrieval. |
| 67–83 | Load Qwen2-VL, build chat template with retrieved images, generate, full pipeline | **Focus + slow down.** End-to-end visual QA. |

---

## ✅ Walk-away checklist

After the notebook, you should be able to say, in your own words:

- [ ] Why text-only RAG fails on image queries, and what multimodal RAG adds.
- [ ] What a **shared embedding space** is and why CLIP's dual encoder creates one.
- [ ] How CLIP is trained (contrastive matching, symmetric loss, the diagonal).
- [ ] The three cross-modal search directions (text→image, image→image, image→text).
- [ ] Why a **vision-language model** is needed for generation (a text LLM can't see images).
- [ ] How **ColPali + Qwen2-VL** answer document questions *without OCR*.
- [ ] The hardware tradeoff between System A (API/CPU-friendly) and System B (big GPU).

---

## 🎯 5-question self-check

Answer these using only this Brief. Answers are hidden below.

1. In one sentence, what property of CLIP makes it possible to retrieve images using a text query?
2. During CLIP training, an N×N similarity matrix is built for a batch. Which entries should be large, and which small?
3. You retrieve the three most relevant images for a question. Why can't you just send them to a standard text LLM like `gpt-4o-mini` (text-only) for the answer?
4. System B answers questions about IKEA manuals *without* running OCR. How does it read the pages instead?
5. Name one concrete advantage and one concrete cost of the ColPali + Qwen2-VL approach versus the CLIP + GPT-4o-Vision approach.

<details>
<summary>Answers</summary>

1. CLIP embeds **text and images into the same (shared) 512-dim vector space**, so a text query's vector can be compared directly to stored image vectors via nearest-neighbor / cosine similarity.
2. The **diagonal** entries (each image with its *true* caption) should be **large**; all **off-diagonal** entries (mismatched image-caption pairs) should be **small**. The symmetric cross-entropy loss enforces exactly this.
3. A text-only LLM **cannot see the images** — it only processes text. You need a **vision-language model** (GPT-4o Vision, Qwen2-VL) that accepts images as input so it can reason about what's actually *in* the retrieved pictures (diagrams, layouts, objects).
4. It treats each **PDF page as an image**: **ColPali** indexes/retrieves page-images by semantic similarity, and **Qwen2-VL** reads those page-images directly to answer — the model "looks at" the page rather than relying on transcribed text.
5. Advantage: **no OCR needed** — it handles diagrams, tables, and visual layouts that OCR mangles (and understands meaning, not just keywords). Cost: it needs a **large GPU (≥24 GB VRAM)** to run Qwen2-VL locally, unlike the lighter CLIP + API approach.

</details>

[🔝 Back to top](#top)
