<a id="top"></a>
# Speech (Audio for GenAI) — Reading Brief

> **Read this ONCE, end to end, before opening the notebook.** Target ~22 min. By the time you reach the notebook, every word will already make sense — you'll be confirming, not learning blind.
>
> **Side reference:** keep [`Speech_Jargon_Card.md`](./Speech_Jargon_Card.md) open in another tab; look up unknown words there.
> **The notebook:** the detailed reference notebook and its shorter classroom twin, both in this folder.

---

## 🎯 30-second TL;DR

**A model never hears sound. It eats a tensor (grid of numbers). Almost every audio bug is a bug in *how that tensor was made* — wrong sample rate, wrong shape, wrong number range.**

The notebook walks the whole path from raw air pressure to a talking agent:

- **Left of the model:** a sound is a 1-D array of amplitude **samples**. Two knobs define it — **sample rate** (how *often* you measured; 16 kHz for speech AI) and **bit depth** (how *precisely* each measurement is stored).
- **The front-end:** the waveform becomes a **log-mel spectrogram** — an `(80, T)` picture of which frequencies are present over time. That image, not the raw wave, is what Whisper reads.
- **ASR (Automatic Speech Recognition):** **Whisper** (`base.en`, ~73M params) turns speech → text, scoring **WER (Word Error Rate) = 0.059** normalized here.
- **TTS (Text-to-Speech):** a **VITS / MMS-TTS** model turns text → speech (the reverse).
- **The #1 bug, everywhere:** **sample-rate mismatch** — tell a model 16 kHz when audio is really 24 kHz and it silently returns garbage.

Burn this in: **"Get the tensor right — mono, `float32` in `[-1, 1]`, at the model's rate — and most audio problems disappear."**

---

## 🗺️ Agenda — what the notebook teaches, in order

1. **Audio is an array of numbers** — the two units: sample rate (time axis) and amplitude/bit depth (value axis).
2. **A sound is a sum of frequencies** — pure tones added; the **FFT (Fast Fourier Transform)** reads back "the recipe".
3. **Sampling & the Nyquist limit** — why you need ≥ 2 samples per cycle, and what **aliasing** sounds like when you don't.
4. **A man's voice vs a woman's** — **pitch (F0)**, **harmonics**, **formants**, the **source–filter** model, **voiced vs unvoiced** (**RMS** + **ZCR**), and a **low-pass filter** (telephone effect).
5. **Bit depth & quantization** — amplitude resolution; the **~6 dB-per-bit SNR** rule, heard from 16 → 2 bits.
6. **Loading audio without footguns** — sample-rate mismatch, stereo→mono, int16→float32, codecs, clipping.
7. **The log-mel spectrogram** — what the model sees: 25 ms window / 10 ms hop / 80 mel bands → 100 columns/sec.
8. **ASR with Whisper** — encoder–decoder, knobs, hallucination + loops, **VAD (Voice Activity Detection)**, **WER**.
9. **Long audio** — chunking with **stride** and timestamps (Whisper only sees 30 s at a time).
10. **TTS** — text → speech with VITS locally + the OpenAI API pattern.
11. **The voice-agent loop** — `mic → VAD → ASR → LLM → TTS → speaker`; latency, barge-in, cascade vs. audio-native.

---

## 🧠 The big idea — one analogy

**Analogy: digital audio traces a wave with dots on graph paper.**

A sound is a smooth wobble of air pressure over time. A computer can't store a smooth curve, so it approximates it with dots:

- **Sample rate** = how many dots *along the time axis* (horizontal detail). More dots/second = you can trace faster wiggles (higher frequencies). Speech AI = 16,000 dots/second (16 kHz).
- **Bit depth** = how many *vertical positions each dot can snap to*. More levels = each dot lands closer to the true height, less rounding error.

Everything else follows from this one picture:

- Too few dots per second and a fast wave *also fits* a slow one — misread as a slow tone (**aliasing**, from the **Nyquist limit**).
- Too few vertical levels and each dot lands slightly off — that rounding error *is* **quantization noise** (the ~6 dB-per-bit rule).
- The model doesn't read the dots directly — it reads a summary of *which frequencies they trace over time* (the **log-mel spectrogram**).

Remember the graph-paper picture and sample rate, bit depth, aliasing, quantization, and spectrograms all fall out — no signal-processing math required.

---

## 📖 Core concept primers

Six primers cover the notebook's heart: **mental model**, plain English, real numbers, why it matters.

### 1. Sample rate & the Nyquist limit

> **🪜 Mental model:** dots along the time axis — you need at least *two dots per wiggle* to know the wiggle was there.

**Sample rate** (`sr`, Hz) is how many times per second you measure the wave. Note: **sound frequency** = how fast the wave wiggles; **sample rate** = how fast you measure it — two different things. The **Nyquist limit** says a sample rate faithfully captures frequencies only up to **half of it**: 16 kHz → 8 kHz max, 44.1 kHz → 22.05 kHz, 8 kHz phone → 4 kHz.

Break it and you get **aliasing**: a tone above Nyquist "folds down" into a fake lower one. The notebook demos it — a 3 kHz tone sampled at 4 kHz (Nyquist 2 kHz) with no filter folds to `|4000 − 3000| = 1000 Hz`, a beep never played.

**Why it matters here.** Sample rate is the #1 footgun: pass the wrong rate and the model doesn't error — it returns garbage or wrong-speed text. The whole "load audio without footguns" section defends against this one mistake.

### 2. FFT — a sound is a sum of frequencies

> **🪜 Mental model:** the FFT is a chef tasting a soup and naming the ingredients — hand it a messy wave, it lists the pure tones inside.

Any sound — a vowel, a chord, an orchestra — is just many **pure tones** (single sine waves) added together. Forward (tones → sum) is easy addition. *Backward* — which tones were in the messy sum? — is exactly what the **FFT (Fast Fourier Transform)** does. The notebook adds three tones (220, 440, 660 Hz), FFTs the mix, and gets three clean spikes back at exactly those frequencies — "the recipe".

**Why it matters here.** The FFT is the engine behind everything visual later: a **spectrum** is one FFT of one chunk; a **spectrogram** is that FFT recomputed per time-slice and stacked. You use it as a tool — the notebook never derives it.

### 3. Voice = source (pitch) + filter (vowel); voiced vs unvoiced

> **🪜 Mental model:** the vocal folds are a buzzing reed (the *note*); the mouth is a shaped tube around it (the *vowel*).

Speech splits two ways. First, the **source–filter** model: the **source** is the vocal folds opening/closing at the **fundamental frequency F0** — that *is* the pitch (men ≈ 90–150 Hz, women ≈ 170–250 Hz, nearly twice as fast). F0 and its multiples (**harmonics**) form an evenly-spaced comb of spikes; wider comb = higher pitch. The **filter** is the mouth/throat shape boosting certain bands (**formants**), deciding *which vowel* (speaker-independent). Raise F0 alone → pitch rises, vowel unchanged.

Second, **voiced** vs **unvoiced**. Voiced sounds (vowels) are periodic — high **RMS (Root Mean Square** = frame loudness), low **ZCR (Zero-Crossing Rate** = how often the wave flips sign). Unvoiced (s, f, sh) are noise-like — high ZCR, no pitch. The notebook uses RMS + ZCR to sort frames and plays each group alone.

**Why it matters here.** The voiced/unvoiced split reappears at **VAD**. And the **low-pass filter** demo (drop everything above 3,400 Hz, the telephone cutoff) shows the high frequencies carried the consonant detail — what a wrong-rate resample throws away.

### 4. Bit depth, quantization & the 6 dB-per-bit rule

> **🪜 Mental model:** dots snapping to graph-paper lines — more lines (bits) = each dot lands closer to the true height.


**Bit depth** sets how precisely each sample's *amplitude* is stored. With *B* bits you get `2^B` levels; every value rounds to the nearest, and that error is audible **quantization noise**. Rule of thumb: **each bit buys ~6 dB of SNR (Signal-to-Noise Ratio** = how much louder the signal is than the noise). The notebook requantizes the real clip and measures it:

| Bits | Levels | SNR (dB) |
|---|---|---|
| 16 | 65,536 | 72.5 (transparent — CD quality) |
| 8 | 256 | 27.8 |
| 4 | 16 | 2.0 |
| 2 | 4 | −13.7 (noise now louder than signal) |

Every 4-bit drop loses ~24 dB — squarely the 6-dB-per-bit rule.

**Why it matters here.** It cleanly separates the two axes: sample rate = *time*, bit depth = *amplitude*, no interaction. It also explains why files ship as `int16` (16-bit, ~96 dB, half the bytes) while models want `float32` in `[-1, 1]`.

### 5. The log-mel spectrogram — what the model actually sees

> **🪜 Mental model:** slide a short window along the audio, ask "what frequencies are in *this* window?", stack the answers as columns — that's the spectrogram.

Models rarely read the raw waveform (16,000 numbers/second is too much). The **front-end** builds a **spectrogram** via the **STFT (Short-Time Fourier Transform)**: chop the audio into short overlapping windows, FFT each. Two refinements make the **log-mel spectrogram** every speech model eats: (1) squash the FFT bins into ~80 perceptual **mel** bands (fine at low frequency, coarse at high — like human hearing), and (2) take a **log** (dB) so quiet detail survives.

The canonical config — **25 ms window / 10 ms hop / 80 mel bands / 0–8 kHz @ 16 kHz** — produces exactly **100 columns/second**. In the notebook the 5.86 s clip becomes an `(80, 586)` image; as a tensor `(1, 80, 586)` = `(batch, mel bands, time frames)`. Whisper's extractor pads it to `(1, 80, 3000)` (its fixed 30 s window).

**Why it matters here.** This `(80, T)` image is *all the encoder reads*. The one knob worth knowing is **window length**: short = sharp in time, blurry in frequency; long = reverse. 25 ms is the compromise — reads pitch, still catches fast consonants.

### 6. ASR (Whisper) + WER + VAD

> **🪜 Mental model:** Whisper = a reader (encoder eyes over the spectrogram) + a typist (decoder writing words one at a time, glancing back at the audio).

**Whisper** is a Transformer **encoder–decoder**: the encoder reads the whole log-mel image; the decoder emits tokens one at a time (**autoregressive**), attending back to the audio. Its task is set by a *token prefix* — `<|transcribe|>` vs `<|translate|>` — not a different model. The notebook runs `whisper-base.en` (~73M params); its output vs the reference is in the headline table below.

You score ASR with **WER (Word Error Rate)**, *not* like text generation. Two famous failure modes: **hallucination** (fed silence, it invents *"Thank you." / "Thanks for watching!"* from training data) and **repetition loops**. The #1 fix for both: **VAD (Voice Activity Detection)** — detect *where* there's speech first (Silero VAD default) and transcribe only those segments.

**Why it matters here.** This is the payoff of the front-end: everything before made the tensor; here the model consumes it. The sample-rate footgun reappears literally — `processor(y16, sampling_rate=16_000)` *must* match the true rate, or the transcript is wrong.

---

## 🔥 The headline experiment — at a glance

Two measurable results: **Whisper's WER** and the **SNR-vs-bit-depth ladder**.

**Whisper ASR accuracy (WER) on the LibriSpeech demo clip:**

| | Value | Note |
|---|---|---|
| Model | `whisper-base.en` (~73M params) | fast classroom default |
| Reference | *"MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES…"* | 16 words, ground truth |
| Hypothesis | *"Mr. Quilter is the apostle of the middle classes…"* | Whisper output |
| **WER (raw)** | **0.176** | punishes "Mr." vs "MISTER", casing, punctuation |
| **WER (normalized)** | **0.059** | ← **report THIS** (lowercase, strip punctuation first) |
| Toy sanity check | 0.222 | 1 sub + 1 del = 2 errors / 9 words |

**Takeaway 1:** raw WER (0.176) vs normalized WER (0.059) — a 3× gap from formatting alone. *Always normalize both sides, or you measure typography, not recognition.*

**Bit-depth → SNR ladder (~6 dB-per-bit, measured):** 16-bit = 72.5 dB → 8-bit = 27.8 → 4-bit = 2.0 → 2-bit = −13.7 dB (noise now louder than signal); each 4 bits ≈ 24 dB lost.

**Takeaway 2:** two independent axes — sample rate = *what frequencies survive*; bit depth = *how clean each sample is*. Confuse them and you'll "fix" the wrong knob.

---

## 🧮 Formulas to memorise

Only three carry their weight for interviews.

### 1. Nyquist sampling rule

```
sample_rate ≥ 2 × highest_frequency_you_want
   (equivalently: max faithful frequency = sample_rate / 2)
```

**In words:** you must sample at least twice per cycle, so a sample rate can only represent frequencies up to *half* of it (the **Nyquist limit**). Anything above folds into a fake lower tone (**aliasing**).

**Worked example (notebook quiz):** to capture a 12 kHz sound you need `2 × 12,000 = 24,000 Hz` = 24 kHz min. A 3 kHz tone sampled at 4 kHz (limit 2 kHz) aliases to `|4000 − 3000| = 1000 Hz`.

### 2. SNR per bit (quantization)

```
SNR (dB)  ≈  6.02 × (number of bits)
   → each bit added/removed changes SNR by ~6 dB
```

**In words:** the signal-to-noise ratio in decibels is roughly six times the bit depth; each bit of amplitude precision buys ~6 dB of headroom over the quantization noise (each bit removed costs ~6 dB).

**Worked example (notebook quiz):** 16-bit → 12-bit = drop 4 bits → lose `4 × 6 ≈ 24 dB` of SNR. (Sample rate untouched — only the amplitude axis changed.)

### 3. Word Error Rate (WER)

```
WER = (S + D + I) / N
```

**In words:** word error rate = (word **S**ubstitutions + **D**eletions + **I**nsertions) ÷ **N** (words in the reference) — computed *after* normalizing case and punctuation. Lower is better; can exceed 100% since insertions count.

**Worked example (notebook quiz):** reference 20 words; output has 1 sub + 2 del + 1 ins → `WER = (1+2+1)/20 = 4/20 = 0.20` = 20%. (Notebook toy example: 2 errors / 9 words = 0.222.)

---

## 🗺️ Notebook reading map — where to spend your attention

| Cells | What it teaches | How to read |
|---|---|---|
| **0–7** | Setup, deps, capability detection, config (`SR = 16000`, ASR/TTS model ids) | **Skim** — ~3 min. Just note the 16 kHz standard and the two model ids. |
| **8–19** | Audio as an array; sample rate vs sound frequency; loading a real 16 kHz clip | **Read normally** — ~8 min. Watch the raw floats in `[-1, 1]`. |
| **20–32** | Sum of frequencies + **FFT**; **Nyquist limit** & **aliasing** (hear the fold-down) | **FOCUS** — ~10 min. Powers everything after. |
| **33–48** | Man vs woman voice: **F0**, harmonics, formants, **voiced/unvoiced** (RMS/ZCR), low-pass; **bit depth** & SNR ladder | **Read carefully** — ~12 min. Note the SNR table. |
| **49–54** | Loading footguns: rate mismatch (chipmunk/slow-mo), stereo→mono, int16→float, `librosa.resample` vs `wav[::2]` | **FOCUS — the #1 bug** — ~8 min. |
| **55–63** | The **log-mel spectrogram**: STFT, mel, log; `(80, 586)`; window trade-off | **Read carefully** — ~8 min. "What the model sees." |
| **64–78** | **Whisper** encoder–decoder, knobs, hallucination + loops, **VAD**, **WER** | **FOCUS — the ASR payoff** — ~12 min. |
| **79–95** | OpenAI transcription/TTS API; chunking + stride; **TTS** (VITS) + API pattern; agent cascade, latency, barge-in | **Read normally** — ~10 min. TTS 24 kHz ≠ ASR 16 kHz. |
| **96** | Recap cheat-sheet, gotcha list, troubleshooting, exercises | **Reference** — bookmark it. |

**Total notebook read:** ~75 min + this brief's ~22 min = **~97 min**, vs. the ~150 min notebook cold.

---

## ✅ Walk-away checklist

After the notebook, you should be able to say in your own words:

- [ ] **Why "a model never hears sound"** — it eats a tensor; most bugs are in how that tensor was made (rate, shape, range).
- [ ] **The two axes of digital audio** — sample rate (time; 16 kHz for speech) vs bit depth (amplitude; ~6 dB per bit). They're independent.
- [ ] **The Nyquist limit and aliasing** — need ≥ 2 samples per cycle; a tone above `sr/2` folds down to `|sr − f|`.
- [ ] **What the FFT does** — decomposes a messy wave into its pure-tone recipe; a spectrogram is the FFT per time-slice, stacked.
- [ ] **What Whisper actually reads** — an `(80, T)` log-mel image, ~100 columns per second, *not* the raw wave.
- [ ] **How ASR is scored** — WER = (S + D + I) / N, on *normalized* text (0.059 vs 0.176 raw here).
- [ ] **The #1 production bug and its fix** — sample-rate mismatch; resample to the model's rate *and* pass it everywhere. And VAD-before-ASR kills most Whisper hallucinations.

If any feel shaky, come back to the relevant primer above.

---

## 🎯 5-question self-check

Answer in your head, then check below. **No peeking.**

1. A loader hands you a mono array of shape `(48000,)` for a clip you *know* is exactly 3 seconds long. What sample rate was it recorded at?
2. A pure 5 kHz tone is sampled at 8 kHz with no anti-alias filter. What false frequency do you hear, and why?
3. You requantize audio from 16-bit down to 12-bit. Roughly how much SNR do you lose, and does the sample rate change?
4. A reference sentence has 20 words; Whisper's output has 1 substitution, 2 deletions, and 1 insertion. What is the WER?
5. In one sentence: your TTS model emits 24 kHz audio and you feed it straight into a 16 kHz ASR model with `sampling_rate=16000`. What goes wrong, and what's the fix?

---

<details>
<summary><b>Click to reveal answers</b></summary>

1. **16 kHz.** Sample rate = samples ÷ seconds = `48000 / 3 = 16000 Hz`. The array length alone means nothing until you divide by duration.
2. **3 kHz.** 5 kHz sits above the 4 kHz Nyquist limit (half of 8 kHz), so it folds down to `|sample_rate − f| = |8000 − 5000| = 3000 Hz` — a lower pitch that was never played (**aliasing**).
3. **~24 dB lost; sample rate is unchanged.** You dropped 4 bits, each worth ~6 dB of SNR, so `4 × 6 ≈ 24 dB`. Bit depth is the amplitude axis only; the time axis (sample rate) is independent.
4. **0.20 (20%).** `WER = (S + D + I) / N = (1 + 2 + 1) / 20 = 4 / 20 = 0.20`. Insertions count too, which is why WER can occasionally exceed 100%.
5. **ASR misreads it — declaring 16 kHz for 24 kHz audio stretches it in time and shifts every frequency, corrupting the model's input.** Nothing fixes it automatically; you must `librosa.resample` the TTS output 24 kHz → 16 kHz *before* feeding it in. (This is the sample-rate mismatch footgun, now spanning the TTS→ASR hand-off.)

</details>

---

[🔝 Back to top](#top) · [→ Jargon Card](./Speech_Jargon_Card.md)
