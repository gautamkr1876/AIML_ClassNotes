<a id="top"></a>
# Video (AI Modalities) — Jargon Card

> **Use this file like a dictionary.** Skim it once (~5 min) before opening the notebook. Then keep it open in a side tab — when you hit an unknown word while reading, look it up here in 20 seconds instead of Googling for 5 minutes.
>
> **Companion:** read [`Video_Reading_Brief.md`](./Video_Reading_Brief.md) FIRST — that's the story. This card is just the dictionary.
> **The notebooks:** `ai_modality_video.ipynb` (the main pipeline) and `open_cv_demo.ipynb` (OpenCV image basics) in this folder.

---

## A

**ASR (Automatic Speech Recognition)** — Turning spoken audio into written text. In this notebook the ASR model is **Whisper** (`whisper-1`), called with `client.audio.transcriptions.create(...)`. It's the tool that converts the clip's dialogue into a searchable, timestamped transcript.

**Aspect ratio** — The width-to-height proportion of an image or frame (e.g., 1920×1080 is 16:9). The notebook's `extract_frames` resizes by shrinking the longer side to `size=512` while keeping the ratio, so a 1920×1080 frame becomes 512×288, not a squashed 512×512.

## B

**Base64** — A way of writing binary data (like a JPEG's bytes) using only plain text characters, so it can be pasted into a JSON request. The notebook's `frame_to_data_url` encodes each frame as Base64 because the chat API sends everything as text. Downside: Base64 makes data ~33% bigger, which is why small frames matter for cost.

**BGR vs RGB** — Two orderings of the three colour channels (Red, Green, Blue). **OpenCV loads images as BGR** (Blue first) while almost everything else (matplotlib, the vision API) expects **RGB**. This mismatch is the notebook's #1 gotcha: forget `cv2.cvtColor(..., COLOR_BGR2RGB)` and your reds and blues swap.

## C

**Channel** — One colour layer of an image. A colour frame has 3 channels (R, G, B); grayscale has 1. Shapes in the notebook read as `(H, W, 3)` for colour and `(H, W)` for gray — the trailing `3` is the channel count.

**Chunk (video chunk)** — A small, self-contained, searchable piece of the video: a `{timestamp, text}` record built either from a **frame caption** (visual route) or a **transcript segment** (audio route). Chunks are what you embed and search over — the foundation of **video RAG**.

**Codec (H.264 / H.265)** — The compression scheme that shrinks video on disk (H.264, a.k.a. AVC; H.265, a.k.a. HEVC). The notebook flags that a 57-second clip is only a few MB on disk but **8.5 GB** once fully decoded to raw pixels — that gap is the codec doing its job.

**Compression** — Storing video in far fewer bytes than the raw pixels would need, by exploiting redundancy between frames. Why the notebook's `bribe.mp4` is small on disk but huge (`n_frames × H × W × 3` bytes) in memory once decoded.

## D

**Data URL** — A single string that embeds a whole file inline, like `data:image/jpeg;base64,/9j/4AA...`. The vision API accepts frames this way. `frame_to_data_url` builds one per frame; the notebook prints their sizes in KB (4–32 KB each).

**Decode** — Turning compressed video bytes back into actual pixel arrays you can process. `cap.read()` decodes one frame at a time. Decoding is the expensive step, which is why the notebook decodes densely only at a tiny 64×64 resolution when it just needs a change signal.

**detail ("low" / "high")** — A vision-API knob controlling how many pixels of your image the model actually looks at. `"low"` uses a small fixed tile (cheap, fewer tokens); `"high"` reads the full resolution (pricier). The notebook sends frames at `detail="low"` because 8 frames add up fast.

**Duration** — How long the clip runs, computed as `n_frames / fps`. In the notebook: `1367 / 23.81 ≈ 57.4 s`.

## E

**Embedding** — Turning a piece of text (a caption or transcript segment) into a list of numbers (a vector) that captures its meaning, so similar meanings land near each other in that number-space. The notebook's `EMBED_MODEL = "text-embedding-3-small"` does this; it's the step that makes **semantic search** over video moments possible.

**extract_frames** — The notebook's own helper that decodes a video, uniformly samples `num_frames` frames, shrinks each to `size`, converts BGR→RGB, and returns the frames **plus their timestamps**. The workhorse of the whole pipeline.

## F

**ffmpeg** — The Swiss-army-knife command-line tool for audio/video. The notebook uses the bundled `imageio_ffmpeg` binary to strip the audio track out of the video (`-vn` = "no video") into a 16 kHz mono WAV, ready for Whisper.

**Frame** — A single still image from the video. A video is just a fast sequence of frames; the notebook's clip has 1367 of them. Everything downstream (sampling, sending to the model) operates on frames.

**Frame sampling** — Picking a small, evenly-spaced subset of frames instead of sending all of them. The notebook samples **8** frames from 1367 via `np.linspace(...)`, because consecutive frames are near-duplicates — sending them all wastes money for almost no extra information.

**FPS (Frames Per Second)** — How many frames play each second. The clip runs at 23.81 fps. The notebook notes movies use ~24 fps while gaming wants 120 fps — higher fps = smoother motion but more frames to store and process.

**Fusion (audio + video / multimodal fusion)** — Feeding the model **both** the timestamped transcript *and* keyframes in one request, so it can reason about what was said and what was shown together. The notebook's fused prompt produces a timestamp-cited summary and even names the movie (*The Naked Gun 2½*).

## G

**Grayscale** — A single-channel, black-and-white image (shape `(H, W)`, no colour). The notebook converts frames to grayscale at 64×64 for cheap change-detection, since colour isn't needed to measure "how much moved."

## H

**H×W×C (shape convention)** — How image/frame arrays are laid out: **H**eight, **W**idth, **C**hannels. A `(288, 512, 3)` frame is 288 rows tall, 512 columns wide, 3 colour channels. Videos add a time axis up front: `(T, H, W, C)`.

## I

**imencode / imwrite** — OpenCV functions to compress an image into a format. `cv2.imencode(".jpg", ...)` compresses to JPEG bytes in memory (used to build data URLs); `cv2.imwrite("x.png", ...)` writes to a file on disk.

**Interpolation** — The math OpenCV uses to invent pixel values when resizing. `INTER_AREA` is best for shrinking; `INTER_CUBIC` for enlarging; `INTER_LINEAR` is the default. The notebook uses `INTER_AREA` because it's downscaling.

## J

**JPEG vs PNG** — Two image file formats. **JPEG** is lossy but ~10× smaller for photos — the notebook uses it (`quality=80`) to send frames cheaply. **PNG** is lossless (bigger); the OpenCV demo writes copies as PNG.

## K

**Keyframe** — A representative sampled frame chosen to stand in for a stretch of video. In the fusion step the notebook sends "a few keyframes for visual context" alongside the transcript — the visual anchors the model reasons over.

## L

**linspace (np.linspace)** — NumPy's "give me N evenly-spaced numbers between A and B" function. The notebook uses it to pick which frame indices to sample: `np.linspace(0, n_frames-1, 8)` → 8 evenly-spread frames across the whole clip.

## M

**MAD (Mean Absolute Difference)** — The average pixel-by-pixel difference between two consecutive frames. **Low MAD** = frames barely changed (safe to skip → sampling works). **A spike in MAD** = a big change, usually a **shot cut**. The notebook computes it with `np.abs(np.diff(small, axis=0)).mean(...)`.

**Metadata (video)** — Facts about the video you can read *without* decoding pixels: fps, frame count, width, height. Cheap to grab via `cap.get(cv2.CAP_PROP_*)`. Step 1 of the pipeline.

**Modality** — A *type* of data / sense the model works with: text, image, audio, video. "AI modalities for video" means teaching a text-and-image model to handle video by decomposing it into modalities it already understands (frames = images, speech = audio→text).

**Mono / 16 kHz** — Audio settings the notebook forces when extracting the WAV: **mono** = one channel (not stereo), **16 kHz** = 16,000 audio samples per second. This is the compact format ASR models like Whisper expect (`-ac 1 -ar 16000`).

**Multimodal** — A model or prompt that handles more than one modality at once (here: text + images, plus audio via transcription). The whole notebook is a multimodal video-understanding pipeline built on top of a vision LLM.

## R

**RAG (Retrieval-Augmented Generation)** — Answer a question by first *retrieving* the most relevant stored chunks, then feeding them to an LLM to *generate* a grounded answer. **Video RAG** = the same idea over video: embed frame-captions and transcript segments, retrieve the moments matching a query, then answer. The pipeline's final destination.

**Redundancy (temporal redundancy)** — The fact that neighbouring video frames look almost identical. This is *why* frame sampling is safe and why video compresses so well. The notebook proves it by showing a low mean frame-to-frame difference (MAD ≈ 3.52 on a 0–255 scale).

**Resolution** — The pixel dimensions of a frame (e.g., 1920×1080). Higher resolution = more visual detail but more visual tokens and higher cost. The notebook down-samples resolution to trade a little detail for a lot of savings.

**Responses API vs Chat Completions API** — Two OpenAI request styles the notebook shows side by side. **Chat Completions** (`client.chat.completions.create`) uses `messages` with `image_url`; the newer **Responses API** (`client.responses.create`) uses `input` with `input_image` and exposes `.output_text` and token `usage`. Same task, different shapes.

## S

**Sampling (uniform / even sampling)** — Choosing frames at equal time intervals across the clip (vs. clustering them). The notebook uses uniform sampling so the 8 frames span the full 57 seconds evenly (t = 0.0, 8.2, 16.4 … 57.4 s).

**Semantic search** — Finding content by *meaning*, not exact keywords, by comparing embedding vectors. The notebook's video RAG uses it: embed the user's question, compare against all chunk vectors, return the closest moments with their timestamps.

**Shot cut / shot boundary** — The moment the camera or scene changes abruptly. Shows up as a **spike** in the MAD change signal. Useful as a natural chunk boundary when splitting long video.

**Segment (transcript segment)** — One timestamped line of transcript: `{start, end, text}`. Whisper returns these when called with `response_format="verbose_json"` and `timestamp_granularities=["segment"]`. The notebook got 49 segments from the 57 s clip.

## T

**Temporal (temporal reasoning / modeling)** — Anything to do with *time / order* in the video. The core challenge: a still frame has no clock, so the notebook **explicitly labels each frame with its timestamp** (`[frame @ 8.2s]`) so the model can reason about what happens *over time*, not just in one frame.

**Timestamp** — The point in time (in seconds) a frame or transcript segment belongs to. The notebook preserves timestamps everywhere because "the model cannot infer time on its own — we have to tell it." They're also the metadata that lets RAG point you back to *when* something happened.

**Token / visual token** — The unit LLMs bill and think in. Text becomes text tokens; images become **visual tokens** (more pixels / higher `detail` = more visual tokens = more cost). The notebook's cost story is entirely about keeping visual-token count down via sampling + resizing + `detail="low"`.

**Transcript** — The full timestamped text of what's spoken, assembled by joining all segments into one block. The notebook feeds this block to the model as `TRANSCRIPT (timestamped):` in the fusion step.

## V

**VideoCapture (cv2.VideoCapture)** — OpenCV's object for opening and reading a video file frame by frame. `cap.read()` returns `(ok, frame)`; `cap.get(cv2.CAP_PROP_*)` reads metadata. Always `cap.release()` when done.

**Vision model / VLM (Vision-Language Model)** — An LLM that accepts images alongside text (here `gpt-5-nano`, or any of gpt-4o / gpt-4.1). It's the "eyes" of the pipeline: you hand it labelled frames and it describes what happens across the clip.

---

[🔝 Back to top](#top) · [→ Reading Brief](./Video_Reading_Brief.md)
