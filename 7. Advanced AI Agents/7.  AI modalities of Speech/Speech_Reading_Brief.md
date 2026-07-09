<a id="top"></a>
# AI Modalities: Speech — Reading Brief

> **Read this ONCE, end to end, before opening the notebook.** Target time: ~20 minutes. By the time you reach the notebook, every word in it will already make sense — you'll be confirming what you already know, not learning blind.
>
> **Side reference:** keep [`Speech_Jargon_Card.md`](./Speech_Jargon_Card.md) open in another tab while reading. When an unknown word appears, look it up there.
> **The notebook:** `AI_Modalities_Speech.ipynb` (~38 cells, many audio/plot demos).
> **Note:** this notebook is demo-and-code-heavy with little prose, so this brief front-loads the *concepts* the demos illustrate.

---

## 🎯 30-second TL;DR

Before an AI can transcribe or generate speech, sound has to become **numbers**. This notebook builds that pipeline bottom-up: a sound wave → a sampled amplitude array → its frequencies (**FFT**) → a **log-mel spectrogram** → a speech model. The one sentence that anchors everything:

> **Audio is just amplitude numbers over time; every speech model first turns those numbers into a log-mel spectrogram, then reads it.**

The first half is classic **DSP** (waves, sampling, frequency, voiced vs unvoiced, quantization, mel spectrograms) — the "how sound becomes model input." The second half is the **modality APIs**: **ASR** (speech→text) with **Whisper** and OpenAI's transcription API, and **TTS** (text→speech) with a local VITS model and OpenAI's TTS API — plus **WER** to measure accuracy.

---

## 🗺️ Agenda — what the notebook teaches, in order

1. **Setup** — imports, capability detection (torch/HF optional), sample rate = 16 kHz.
2. **What sound is** — build sine waves at different Hz; combine them; hear them.
3. **Sampling** — a wave becomes an array; `samples = SR × seconds`; zoom to see individual samples.
4. **Frequencies** — sum pure tones into a chord; **FFT** reveals the component frequencies.
5. **Pitch & voice** — F0 + harmonics as a spectral comb; male vs female voice spacing.
6. **Voiced vs unvoiced** — **RMS** (loudness) + **ZCR** (zero-crossings) to split vowels from hiss.
7. **Quantization** — fewer bits → rising noise; SNR ≈ 6 dB/bit.
8. **Mel spectrogram** — the raw-audio → log-mel front-end every speech model uses.
9. **ASR** — Whisper (local) transcribes a clip; OpenAI `gpt-4o-transcribe` via API.
10. **TTS** — local MMS-TTS (VITS) and OpenAI `gpt-4o-mini-tts` synthesize speech.
11. **WER** — Word Error Rate to score transcription quality.

---

## 🧠 The big idea — sound becomes a picture the model reads

A neural network can't hear. It consumes numbers. So the entire job of the "front-end" is to convert a pressure wave in the air into a numeric form that preserves what matters for speech — and it turns out the best form is a **picture**: a spectrogram (frequency on one axis, time on the other, brightness = energy).

**The transferable analogy: sheet music.** A microphone captures a squiggly pressure wave — hard to read directly. A spectrogram is that performance rewritten as sheet music: time flows left-to-right, pitch stacks bottom-to-top, and how loud each note is shows as brightness. Just as a musician reads structure off a score far more easily than off a raw waveform, a speech model reads a **log-mel spectrogram** far more easily than raw samples. The "mel" part warps the pitch axis to match human hearing (we hear fine detail in low frequencies, coarser in high), and the "log" part scales loudness the way ears do. So the pipeline is always:

> **raw wave → short overlapping frames → frequency energy (FFT) → mel compression → log scale → log-mel spectrogram → model encoder.**

Once you see that a `.mp3` is "amplitude numbers," a spectrogram is "those numbers reorganized into a hearing-friendly picture," and Whisper is "a model that reads that picture and writes text," the whole notebook — and speech AI generally — clicks into place.

---

## 📖 Core concept primers

Five primers cover the heart of the notebook. Each has a **mental model**, plain-English meaning, notebook specifics, and why it matters.

### 1. Sound → samples (sampling rate)

> **🪜 Mental model:** a flip-book. The continuous wave is smooth motion; sampling takes snapshots fast enough that playback looks (sounds) continuous.

A sound is a pressure wave; **frequency (Hz)** is how fast it vibrates (pitch), **amplitude** is how big the swing is (loudness). To digitize it, we measure the amplitude many times per second — the **sampling rate**. The notebook standardizes on **16 kHz** (the speech-model convention), so a clip is a 1-D array where `length = SR × seconds` (3 s → 48,000 samples). Zooming to 30 ms shows the individual **samples** as dots. **Why it matters here:** this is the foundational knowledge-check — given a `(48000,)` array for a 3-second clip, the sample rate is 16 kHz. Everything downstream operates on this array.

### 2. Frequencies via FFT (spectrum, pitch, harmonics)

> **🪜 Mental model:** un-mixing paint. A wave is many colors (frequencies) blended into one; the FFT separates the blend back into its component colors and how much of each.

Any complex sound is a sum of simple sine waves. The notebook builds a "chord" by adding pure tones (220 + 440 + 660 Hz) into one array, then uses the **FFT** (`np.fft.rfft`) to recover the **spectrum** — energy at each frequency, showing spikes exactly at the input tones. For a human voice, the spectrum is a **comb**: the lowest spike is **F0** (pitch) and the rest are **harmonics** (multiples of F0). Wider comb spacing = higher pitch — the notebook contrasts a male vs female voice this way. **Why it matters here:** FFT is the bridge from "wiggles over time" to "which pitches are present," and it's the first step of building a spectrogram.

### 3. Voiced vs unvoiced (RMS + ZCR)

> **🪜 Mental model:** humming vs shushing. "Ahh" is a steady periodic hum (voiced); "sss" is turbulent noise (unvoiced). Your ear knows instantly; RMS+ZCR lets code know too.

Speech alternates between **voiced** sounds (vowels — periodic, so **loud** = high RMS and **few** zero-crossings = low ZCR) and **unvoiced** sounds (hiss like "s"/"f" — quieter, **many** zero-crossings = high ZCR). The notebook cuts audio into **frames** (1024 samples, 256 hop), computes **RMS** (per-frame energy) and **ZCR** (sign-flip rate), and separates the two groups — even playing each group alone. **Why it matters here:** it builds intuition for what distinguishes speech sounds, and the notebook is explicit that this RMS+ZCR split is a *teaching heuristic* — production systems use **VAD (Voice Activity Detection)**.

### 4. Quantization & the mel spectrogram

> **🪜 Mental model:** quantization = rounding to price tiers (fewer tiers, coarser prices, more error); mel spectrogram = the hearing-adjusted sheet music.

**Quantization** rounds each amplitude sample to one of `2^bits` levels; the notebook drops 16→8→4→2 bits and you *hear* the noise floor rise, with **SNR climbing ~6 dB per bit**. Then it builds the payoff: the **log-mel spectrogram** — frame the audio, FFT each frame, warp frequencies to the **mel** scale (80 bands, human-hearing-spaced), and take the **log (dB)** of energy. Whisper's exact front-end: 80 mels, 25 ms window, 10 ms hop → ~100 frames/sec. **Why it matters here:** the log-mel is *the* input representation for modern speech models — the notebook literally shows its shape is `(80 mel bands, time frames)`, which is what Whisper's encoder ingests.

### 5. The modalities: ASR and TTS (local + API)

> **🪜 Mental model:** two directions on one road. **ASR** drives sound → text; **TTS** drives text → sound.

**ASR (speech→text):** the notebook loads **Whisper** (`whisper-base.en` for speed; `large-v3-turbo` as the production target), feeds a clip's log-mel features to `model.generate`, and decodes the transcript — then does the same via OpenAI's `gpt-4o-transcribe` API. **TTS (text→speech):** it synthesizes audio locally with **MMS-TTS** (a lightweight **VITS** model, CPU-only) and via OpenAI's `gpt-4o-mini-tts` (voice "nova"). Quality is scored with **WER (Word Error Rate)** — the fraction of words the transcription gets wrong. **Why it matters here:** this is the practical payoff — after understanding the front-end, you see that using speech AI is often a few API/model calls, and you know *what* those calls are consuming (log-mel features) and *how* to measure them (WER).

---

## 🔥 The speech pipeline & tools — at a glance

**The front-end (always the same):**
```
raw wave → frames → FFT (frequency energy) → mel bands (80) → log (dB) → log-mel spectrogram → model
```

**The models/tools used:**

| Task | Local | API | Metric |
|---|---|---|---|
| **ASR** (speech→text) | Whisper (`whisper-base.en` → `large-v3-turbo`) | OpenAI `gpt-4o-transcribe` | **WER** |
| **TTS** (text→speech) | MMS-TTS / VITS (`facebook/mms-tts-eng`, CPU) | OpenAI `gpt-4o-mini-tts` (voice "nova") | — |

**Key numbers:** sample rate **16 kHz**; Whisper front-end **80 mels, 25 ms window, 10 ms hop, ~100 frames/sec**; quantization **~6 dB SNR per bit**.

---

## 🧮 The two relationships to internalize

```
samples = sampling_rate × duration_seconds       # 16000 × 3 = 48000
SNR(quantized) ≈ 6 dB × bits                      # each extra bit ≈ +6 dB cleaner
```

**Word-by-word translation:** (1) "the length of an audio array equals how many samples per second times how many seconds" — so array shape reveals the sample rate. (2) "every bit of quantization precision buys about 6 more decibels of signal-over-noise" — halving the levels (one fewer bit) makes it ~6 dB noisier. **Worked reading:** a `(48000,)` mono array you know is 3 s → `48000 / 3 = 16000` Hz. Dropping 16-bit audio to 8-bit removes 8 bits ≈ **~48 dB** of SNR — which is exactly why the 2-bit demo sounds so gritty.

---

## 🗺️ Notebook reading map

| Cells | What it teaches | How to read |
|---|---|---|
| 0–2 | Imports, capability detection, config (SR=16 kHz, Whisper/TTS ids) | **Skim.** |
| 3–5 | Build sine waves at various Hz; combine; listen | **Focus.** "Sound = wave = numbers." |
| 6–10 | Sampling; wave → array; zoom to samples; SR knowledge-check | **Focus.** |
| 11–17 | Sum tones → chord; **FFT** spectrum; F0/harmonics; male vs female pitch | **Focus.** FFT is the key tool. |
| 18–22 | Voiced vs unvoiced via **RMS + ZCR** | **Read.** Note it's a heuristic (real = VAD). |
| 24 | **Quantization** demo; SNR ≈ 6 dB/bit | **Read.** |
| 25–28 | The front-end pipeline; **log-mel spectrogram** (80 mels, 25 ms/10 ms) | **Focus + slow down.** The model's input. |
| 29–31 | **Whisper** ASR: log-mel features → transcript | **Focus.** |
| 32–36 | OpenAI APIs: `gpt-4o-mini-tts` (TTS) + `gpt-4o-transcribe` (ASR) | **Read.** |
| 37 | Captions/subtitles; **WER** note | **Read.** |

---

## ✅ Walk-away checklist

After the notebook, you should be able to say, in your own words:

- [ ] Why audio is "just amplitude numbers over time," and how sample rate relates array length to duration.
- [ ] What the **FFT** gives you and how pitch (**F0** + harmonics) appears in a spectrum.
- [ ] How **RMS + ZCR** separate voiced from unvoiced (and that real systems use VAD).
- [ ] What **quantization** costs (~6 dB SNR per bit).
- [ ] The raw-wave → **log-mel spectrogram** front-end, and why it's hearing-adjusted.
- [ ] What **ASR** and **TTS** are, which models/APIs do each, and that **WER** scores ASR.

---

## 🎯 5-question self-check

Answer these using only this Brief. Answers are hidden below.

1. A loader hands you a mono array of shape `(48000,)` for a clip you know is 3 seconds long. What was the sample rate?
2. You FFT a short vowel from a voice and see an evenly spaced comb of spikes. What does the lowest spike represent, and what does wider spacing between spikes mean?
3. A frame of audio is loud (high RMS) with few zero-crossings (low ZCR). Voiced or unvoiced — and give an example sound.
4. In one line, what is the input a Whisper-style model's encoder actually consumes — and name two of its front-end settings.
5. Which metric scores an ASR transcription, and does a higher or lower value mean better?

<details>
<summary>Answers</summary>

1. **16 kHz** — `48000 samples / 3 seconds = 16000 samples per second`.
2. The lowest spike is **F0 (the fundamental frequency = pitch)**; the others are **harmonics**. **Wider spacing = higher pitch** (higher F0) — e.g., a typical female voice's comb is spaced wider than a male voice's.
3. **Voiced** — periodic sounds like vowels ("ahh") are loud and cross zero few times. (Unvoiced sounds like "sss"/"fff" are quieter with many zero-crossings.)
4. A **log-mel spectrogram** — shape `(mel bands, time frames)`. Two settings: **80 mel bands**, **25 ms window**, **10 ms hop** (~100 frames/sec). (Any two.)
5. **WER (Word Error Rate)** — **lower is better** (fewer word errors vs the reference).

</details>

[🔝 Back to top](#top)
