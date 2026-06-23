<a id="top"></a>
# Multimodal RAG — Jargon Card

> **Use this file like a dictionary.** Skim it once (~6 min) before opening the notebook. Then keep it open in a side tab — when you hit an unknown word while reading, look it up here in 20 seconds instead of Googling for 5 minutes.
>
> **Companion:** read [`MultimodalRAG_Reading_Brief.md`](./MultimodalRAG_Reading_Brief.md) FIRST. This card is just the dictionary.
> **The notebook:** `L10 (Jun 16th)_ Multimodal RAG.ipynb` in this folder.
> **Sibling:** builds on `../8. Retrieval-Augmented Generation (RAG) Introduction/` and `../9. Embeddings Deep Dive/`.

---

## B

**byaldi** — A Python library that wraps **ColPali** behind a simple API (`RAGMultiModalModel.from_pretrained(...)`, `.index(...)`, `.search(...)`). The notebook uses it for the document-retrieval half so you don't touch ColPali's internals.

**base64** — A way to encode binary data (like an image) as a text string. The notebook converts retrieved PIL images to base64 so they can be sent inside a **GPT-4o Vision** API request (which expects text/JSON, not raw image bytes).

## C

**ChromaDB** — The open-source **vector database** used to store CLIP embeddings and run cosine-similarity search. The notebook creates two collections (`landmark_texts`, `landmark_images`) with `metadata={"hnsw:space": "cosine"}`.

**CLIP (Contrastive Language-Image Pre-training)** — OpenAI's 2021 model trained on ~400 million image-text pairs. Its breakthrough: it puts **text and images in the same vector space**, so you can compare a sentence to a picture directly. The foundation of the whole first half of the notebook. The variant used is `ViT-B/32`, which outputs **512-dimensional** vectors.

**ColPali** — A document-retrieval model (`vidore/colpali-v1.2`, from the ViDoRe benchmark) that treats each **PDF page as an image** and enables semantic search over it — no OCR needed. Used in the notebook's second half to index IKEA manuals.

**Contrastive learning** — How CLIP is trained: instead of predicting an exact caption (hard), it answers "given N images and N captions, which caption goes with which image?" (a matching task). It pulls **matching** image-text pairs together in vector space and pushes **non-matching** pairs apart.

**Cosine similarity** — Similarity by the angle between two vectors. CLIP embeddings are L2-normalized, so a dot product between them *is* cosine similarity. The notebook's text-image similarity heatmap uses it.

**Cross-modal search** — Searching one modality with another: text→image, image→image, image→text. The superpower a shared embedding space unlocks.

## D

**Dual encoder** — CLIP's architecture: a separate **image encoder** and **text encoder**, each producing a vector of the *same* dimension (512). Trained jointly so their outputs are comparable. (Same family as the bi-encoder idea from the Embeddings lecture, but across two modalities.)

## E

**Embedding** — A fixed-length vector capturing meaning. Here, both an image and a caption become 512-dim vectors; similar concepts land near each other regardless of whether they came from pixels or words.

## G

**GPT-4o Vision** — OpenAI's multimodal LLM (`gpt-4o`) that accepts **both text and images** as input and reasons over them. The notebook's *generation* step: retrieved images + text context are packed into one prompt and GPT-4o Vision writes the answer.

## H

**HNSW** — The approximate-nearest-neighbor graph index ChromaDB uses for fast vector search (covered in the RAG-Intro lecture). Here it's set to cosine space.

## I

**Indexing** — Pre-computing embeddings for every item (image, doc page) and storing them so queries are fast. ColPali's `.index()` embeds each PDF page; ChromaDB's `.add()` stores CLIP vectors.

## M

**Modality** — A type of data: text, image, audio, video. "Multimodal" = handling more than one. The whole point of this lecture is searching/answering *across* modalities.

**Multimodal RAG** — RAG where the knowledge base and/or query include images, not just text. Same Retrieve→Augment→Generate loop, but retrieval uses a multimodal embedder (CLIP) or document-image retriever (ColPali), and generation uses a vision LLM (GPT-4o Vision, Qwen2-VL).

## O

**OCR (Optical Character Recognition)** — Extracting text from an image of text. The notebook's key contrast: ColPali + a vision LLM **skip OCR entirely** by treating the page as an image the model reads directly — robust to diagrams, tables, and layouts OCR mangles.

## P

**PIL (Pillow) Image** — Python's standard in-memory image object. The notebook downloads landmark photos and converts PDF pages into PIL images before embedding or sending to a model.

**pdf2image** — Library (`convert_from_path`) that turns each PDF page into a PIL image. Needed because ColPali and Qwen2-VL work on images, not raw PDFs. Relies on the `poppler-utils` system package.

## Q

**Qwen2-VL** — An open-source vision-language model (`Qwen/Qwen2-VL-7B-Instruct`) that answers questions about images. The notebook's *generation* model for the ColPali/IKEA half. Loaded in `bfloat16`; needs a big GPU (≥24 GB VRAM).

## S

**Shared embedding space** — The single vector space where CLIP places both text and images, so "a photo of the Eiffel Tower" (text) and an actual Eiffel Tower image land at nearly the same coordinates. This is what makes cross-modal search possible.

**Symmetric (contrastive) loss** — CLIP's training loss: cross-entropy computed *both* image→text and text→image, then averaged: `(loss_i2t + loss_t2i) / 2`. The correct pairs sit on the diagonal of the similarity matrix; the loss rewards making the diagonal large and off-diagonal small.

## T

**Temperature (CLIP)** — A learnable scaling factor (initialized to 0.07 in the original CLIP) multiplied into the similarity scores before the softmax. It controls how sharp the matching distribution is. (Different from LLM sampling temperature.)

## V

**Vision-Language Model (VLM)** — Any model that jointly understands images and text and can generate language about them (GPT-4o Vision, Qwen2-VL). The "generator" in multimodal RAG.

**ViT-B/32** — The specific CLIP image-encoder backbone used: a Vision Transformer, "Base" size, 32×32 patches. Outputs 512-dim image embeddings.

**ViDoRe** — The "Visual Document Retrieval" benchmark that ColPali was built/evaluated for.

[🔝 Back to top](#top)
