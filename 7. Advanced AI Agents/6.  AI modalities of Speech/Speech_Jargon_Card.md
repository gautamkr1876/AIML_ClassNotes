<a id="top"></a>
# Speech (Audio for GenAI) — Jargon Card

> **Use this file like a dictionary.** Skim it once (~5 min), then keep it open in a side tab — when an unknown word appears in the notebook, look it up here in 20 seconds.
>
> **Companion:** read [`Speech_Reading_Brief.md`](./Speech_Reading_Brief.md) FIRST for the big picture, the analogy, and the formulas. This card is just the dictionary.
>
> **The one idea behind every entry:** a model never "hears" sound — it eats a **tensor** (grid of numbers). Almost every audio bug is a bug in *how that tensor was made*: wrong sample rate, wrong shape, wrong number range.

---

## A

**Aliasing** — A false, lower-pitched tone that appears when you sample too slowly (fewer than 2 samples per cycle): the fast tone "folds down" and is mistaken for a slow one. In the notebook a 3 kHz tone sampled at 4 kHz folds to `|4000 − 3000| = 1000 Hz`. This is why you resample with a filter, never with `wav[::2]`.

**Amplitude** — How far the wave swings up/down at an instant, i.e. how loud it is. Models expect samples as `float32` in `[-1, 1]`. The amplitude axis is set by **bit depth**, separate from the time axis (**sample rate**).

**Anti-alias filter** — Removes frequencies too high to survive the new (lower) sample rate *before* dropping samples, so they can't fold down into fake low tones (**aliasing**). `librosa.resample` does this automatically; naive slicing (`wav[::k]`) skips it and buzzes.

**ASR (Automatic Speech Recognition)** — Turning speech audio *into* text (audio → text); Whisper is the workhorse here. **Twin to disambiguate:** ASR is the opposite of **TTS** (Text-to-Speech, text → audio). Scored by **WER**. (**Autoregressive decoding** = how Whisper writes: one token at a time, each step reading the audio + all prior tokens; one bad token can cascade into a **repetition loop**.)

## B

**Barge-in (interruption)** — When the user starts talking while the agent is still speaking, and the system must detect it (via **VAD**) and instantly stop the **TTS**. Handling it separates a toy agent from a usable one.

**Bit depth** — How many bits store *each* sample's amplitude. With *B* bits you get `2^B` levels (16-bit → 65,536) and every value rounds to the nearest. **Twin to disambiguate:** bit depth = the *amplitude* axis (height precision per dot); **sample rate** = the *time* axis (dots per second). Independent.

## C

**Cascade (voice-agent loop)** — The standard voice-product pipeline: `mic → VAD → ASR → LLM → TTS → speaker`. **Twin to disambiguate:** the cascade chains separate models per stage; an **audio-native multimodal LLM** does it all in one model (lower latency, less control).

**Chunking (long audio)** — Splitting audio longer than 30 s into windows Whisper can handle (it was only trained on 30 s at a time). The HF pipeline uses `chunk_length_s=30` + `stride_length_s=5` so words on a seam aren't cut in half. See **stride**. (**Compression-ratio / no-speech thresholds** = Whisper knobs that drop suspiciously repetitive or silent segments — levers against hallucination.)

## D

**Diarization** — Answering "*who* spoke *when*". Whisper tells you *what*, not *who*, so for meetings you add a diarization model (`pyannote.audio` is standard) and align its speaker turns with Whisper's timestamps. (**dB / decibel** = a logarithmic loudness/quality ratio unit, used here for **SNR** and the ~6 dB-per-bit rule.)

## E

**Encoder–decoder (Whisper architecture)** — A two-part Transformer: the **encoder** reads the whole log-mel image; the **decoder** writes tokens one at a time, attending back to the audio. Same weights do transcribe *and* translate — only a token prefix changes the task. (**Endpointing** = deciding when the user stopped, via **VAD** watching for silence.)

## F

**F0 (fundamental frequency)** — The base frequency at which the vocal folds open and close — this *is* the pitch you hear. Men ≈ 90–150 Hz, women ≈ 170–250 Hz (nearly twice as fast). Its whole-number multiples are the **harmonics**.

**FFT (Fast Fourier Transform)** — A fast algorithm that takes a messy wave and tells you *which pure tones it's made of, and how loud each is* ("find the recipe"). Feed it a chord and it returns spikes exactly at the notes played. Used as a tool, never derived.

**Formant** — A frequency band the mouth/throat shape boosts, deciding *which vowel* you're saying (roughly speaker-independent). In the **source–filter model** formants are the "filter"; **F0** is the "source". Change F0 → pitch changes, vowel stays; change formants → vowel changes.

**Front-end** — Everything *before* the model: turning a 1-D pressure wave into the tensor it reads (resample → mono → log-mel). Get it right and most audio bugs vanish.

## H

**Hallucination (Whisper)** — Fed silence/music/noise, Whisper confidently invents text it saw often in training — *"Thank you."*, *"Thanks for watching!"*, *"♪"*. The #1 fix: run **VAD** first, transcribe only real speech.

**Harmonics** — The whole-number multiples of **F0** (2×F0, 3×F0, …) that form a "comb" of evenly spaced spikes in a voiced spectrum. The spacing *is* the pitch — a higher-F0 voice has a wider comb.

**Hop length** — How far the window slides between spectrogram columns. At 16 kHz a 160-sample hop = 10 ms → exactly **100 columns per second**. Smaller hop = more time detail.

## L

**LibriSpeech** — The public read-speech dataset the demo clip comes from: 5.86 s, 93,680 samples @ 16 kHz, reference *"MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES…"*.

**Log-mel spectrogram** — The picture Whisper actually eats: a spectrogram squashed into ~80 perceptual **mel** bands, then log-scaled (dB) so quiet detail survives. Shape `(80, T)`, T ≈ 100 × seconds. **Twin to disambiguate:** a plain **spectrogram** uses raw linear FFT bins; a **log-mel** compresses them to human hearing and takes a log.

**Low-pass filter** — "Remove the high frequencies": FFT, zero every bin above a cutoff, inverse-FFT back. The notebook low-passes voices at 3,400 Hz (the telephone cutoff) — why phone calls sound muffled and "s" vs "f" blur.

## M

**Mel scale** — A frequency scale bent to match human hearing: fine at low frequencies, coarse at high (100→200 Hz feels bigger than 7100→7200 Hz). Squashing FFT bins into ~80 mel bands makes a **mel-spectrogram**.

**MMS-TTS / VITS** — The light open TTS model the notebook runs locally (`facebook/mms-tts-eng`). **VITS** is its architecture — one end-to-end model going straight from text to waveform, small enough for CPU with no API key.

**Mono vs stereo** — Mono = one channel `(samples,)`; stereo = two `(samples, 2)`. Models want mono. **Average** the channels (`stereo.mean(axis=1)`) — don't drop one, or you lose half the sound.

## N

**Normalization (WER)** — Cleaning up case and punctuation on *both* the reference and the hypothesis before computing **WER**, so you measure recognition, not formatting. In the notebook, raw WER on the Whisper output is 0.176 but *normalized* WER is 0.059 — you always report the normalized number. (Separately, **amplitude normalization** = scaling a waveform so its peak sits near ±1 without **clipping** into distortion.)

**Nyquist limit** — The highest frequency a sample rate can faithfully capture: exactly **half the sample rate**. 16 kHz → 8 kHz max; 44.1 kHz → 22.05 kHz. Anything above it, unfiltered, becomes **aliasing**.

## P

**Pure tone** — A single frequency, one clean sine wave: `sin(2π·f·t)`. Any real sound is a *sum* of many pure tones of different frequencies and loudnesses — the idea that makes spectrograms make sense. (**Pitch** = the perceived highness of a voice, set by the **F0**.)

## Q

**Quantization** — Rounding each amplitude value to the nearest of the `2^B` levels set by **bit depth**. The error is audible **quantization noise**; fewer bits → louder noise floor (heard in the notebook from 16 → 2 bits).

**Quantization noise** — The error `(quantized − original)` from rounding to finite levels. Every bit removed raises it ~6 dB (lowers **SNR** ~6 dB).

## R

**Resampling** — Converting between sample rates (e.g. 44.1 kHz → 16 kHz) *properly*, with an **anti-alias filter**. Use `librosa.resample`, never `wav[::k]`. Golden rule: resample to the model's rate **and** pass that rate everywhere.

**RMS (Root Mean Square)** — A frame's loudness: square the samples, average, square-root. High = loud/energetic, low = silence/weak. Paired with **ZCR** to split **voiced** from **unvoiced** speech.

## S

**Sample** — One measurement of the wave's height at one instant; a clip is a 1-D array of them. At 16 kHz, 30 ms = 480 samples, and the model reads every one.

**Sample rate (sampling rate, `sr`)** — How *often* you measure the wave, in Hz. Phone 8k · **speech AI 16k** · TTS 24k · music 44.1k/48k. **Twin to disambiguate:** sample rate = the *time* axis (dots/sec); **bit depth** = the *amplitude* axis. The #1 footgun — a wrong rate returns garbage without erroring.

**Sample-rate mismatch** — The most common audio bug: telling a model an array is 16 kHz when it's really 44.1 kHz (or vice-versa). Silently misread — wrong speed, wrong pitch, wrong transcript. Heard in the notebook as "chipmunk" (too fast) and "slow-mo" (too slow).

**Silero VAD** — The production-default **VAD** model: tiny, MIT-licensed, runs on CPU, loaded via `torch.hub`. Learned VAD beats the toy RMS/ZCR heuristics on real audio.

**SNR (Signal-to-Noise Ratio)** — How much stronger the signal is than the noise, in dB; higher = cleaner. Governed by the **~6 dB per bit** rule: the notebook measures 72.5 dB @ 16-bit, 27.8 @ 8-bit, 2.0 @ 4-bit, −13.7 @ 2-bit (noise now louder than signal).

**Source–filter model** — The two-part voice model: **source** = vocal folds vibrating at **F0** (sets pitch), **filter** = mouth/throat shape boosting **formants** (sets the vowel). Move source → pitch changes; move filter → vowel changes.

**Spectrogram** — A picture of *which frequencies are present over time*: slide a short window along the audio, **FFT** each, stack as vertical columns. **Twin to disambiguate:** a plain spectrogram uses raw linear FFT bins; a **log-mel spectrogram** (Whisper's input) compresses to ~80 mel bands and takes a log.

**STFT (Short-Time Fourier Transform)** — The mechanism behind a spectrogram: chop audio into short overlapping windows and FFT each. (A **spectrum** = one such FFT, a single strength-vs-frequency slice; a spectrogram is many stacked.) The **window length** trades off — short = sharp in time, blurry in frequency; long = the reverse. 25 ms is the speech sweet spot.

**Stride (stride_length_s)** — In long-audio chunking, how much neighbouring 30 s windows *overlap* (e.g. 5 s), so a word on a seam appears whole in the next chunk (duplicate then reconciled). Not the CNN stride.

## T

**TTS (Text-to-Speech)** — Turning text *into* speech audio (text → audio); the notebook uses **VITS/MMS-TTS** locally plus the OpenAI API pattern. **Twin to disambiguate:** TTS is the reverse of **ASR**. Gotcha: TTS often outputs 24 kHz — resample before feeding a 16 kHz ASR model.

**Temperature (Whisper)** — Randomness in decoding. Keep it at 0; Whisper only bumps it up (0 → 0.2 → …) as a *fallback* when a segment looks degenerate. Higher temperature makes hallucination and loops more likely. (**Timestamps** = per-segment start/end times Whisper can emit for subtitles and long-audio chunking; a **token** = one word-piece the decoder emits at a time.)

## V

**VAD (Voice Activity Detection)** — Detecting *where* there is speech before transcribing. The standard cheap pre-filter: gating ASR on VAD removes most Whisper **hallucinations** and handles **endpointing** in a live agent. Default: **Silero VAD**. (**VITS** = the end-to-end architecture behind the notebook's local TTS; see **MMS-TTS / VITS**.)

**Voiced vs unvoiced** — The two kinds of speech sound. **Voiced** (vowels) are periodic — have an **F0**, clear harmonics, high **RMS**, low **ZCR** (buzzy, carries pitch). **Unvoiced** (s, f, sh) are noise-like — no pitch, high **ZCR** (hissy). The RMS+ZCR split is a teaching heuristic; real systems use **VAD**.

## W

**Waveform** — The raw 1-D array of amplitude samples, `(samples,)` `float32` in `[-1, 1]`. The rawest representation; spectrogram and tokens are derived from it.

**WER (Word Error Rate)** — The standard ASR score: `(Substitutions + Deletions + Insertions) / reference words`, after **normalization**. Lower is better; can exceed 100% since insertions count. Toy example = 0.222 (2/9); real Whisper = 0.059 normalized.

**Whisper** — OpenAI's Transformer **encoder–decoder** ASR model; the notebook runs `whisper-base.en` (~73M params). Its task (transcribe vs. translate, language) is set by a *token prefix*, not a different model. Failure modes: hallucination on silence, repetition loops.

**Window length (`win_length`)** — The size of each **STFT** analysis window (400 samples = 25 ms at 16 kHz). The one front-end knob worth knowing: short = better time / worse frequency resolution; long = the reverse.

## Z

**ZCR (Zero-Crossing Rate)** — How often the waveform flips sign inside a frame. Low = smooth/periodic (voiced vowels); high = noisy/hissy (unvoiced consonants). Paired with **RMS** to split voiced from unvoiced.

---

[🔝 Back to top](#top) · [→ Reading Brief](./Speech_Reading_Brief.md)
