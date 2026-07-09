<a id="top"></a>
# AI Modalities: Speech — Jargon Card

> **Use this file like a dictionary.** Skim it once (~5 min) before opening the notebook. Then keep it open in a side tab — when you hit an unknown word while reading, look it up here in 20 seconds instead of Googling for 5 minutes.
>
> **Companion:** read [`Speech_Reading_Brief.md`](./Speech_Reading_Brief.md) FIRST. This card is just the dictionary.
> **The notebook:** `AI_Modalities_Speech.ipynb` in this folder.

---

## A

**Amplitude** — How large a sound wave's pressure swing is at a moment — loudness. A digital audio clip is just an array of amplitude samples over time, each in `[-1, 1]` for float audio.

**ASR (Automatic Speech Recognition)** — Speech → text ("transcription"). The notebook uses **Whisper** locally and OpenAI's `gpt-4o-transcribe` via API. Measured with **WER**.

**Amplitude array** — The raw waveform: a 1-D NumPy array where each number is one pressure **sample**. A mono 3-second clip at 16 kHz is shape `(48000,)`.

## D

**dB (decibel)** — A logarithmic loudness/ratio unit. Used here for the log-mel spectrogram (`power_to_db`) and for **SNR**. Rule from the notebook: quantization SNR rises ~**6 dB per bit**.

**DSP (Digital Signal Processing)** — Classic (non-neural) audio math: framing, FFT, RMS, ZCR, spectrograms. The notebook's first half is DSP; it needs only numpy + librosa.

## F

**FFT / rfft (Fast Fourier Transform)** — The algorithm that decomposes a waveform into the frequencies it's made of. `np.fft.rfft` gives the **spectrum** (energy per frequency). Turns "wiggles over time" into "how much of each pitch."

**F0 (fundamental frequency) / pitch** — The lowest, base frequency of a voiced sound — perceived as pitch. In a voiced spectrum it shows as a **comb** of evenly spaced spikes (F0 + harmonics); wider spacing = higher pitch (the notebook contrasts a male vs female voice).

**Frame** — A short slice of audio (e.g., 1024 samples ≈ 64 ms) that DSP analyzes as a unit. Audio is cut into overlapping frames before FFT/RMS/ZCR. `librosa.util.frame` does the cutting.

**Frequency (Hz)** — Cycles per second of a wave — perceived as pitch. 100 Hz = deep, 1000 Hz = sharp, ~15–18 kHz nears the upper limit of human hearing. A pure tone is one sine wave at one frequency.

## H

**Harmonics** — Integer multiples of F0 that accompany a voiced sound, giving it timbre. The evenly spaced spikes above F0 in the spectrum.

**Hop length** — How far the frame window slides between successive frames (e.g., 160 samples = 10 ms at 16 kHz). Smaller hop = more time frames = finer time resolution. Sets the spectrogram's ~100 frames/sec.

**Hz (Hertz)** — Unit of frequency (cycles per second). See **Frequency**.

## L

**librosa** — The Python audio-analysis library doing the DSP here: framing, `melspectrogram`, `power_to_db`, `zero_crossing_rate`, `frames_to_time`.

**Log-mel spectrogram** — The mel spectrogram with amplitudes put on a **log (dB)** scale (`power_to_db`) — matching how humans perceive loudness. This is the exact input a speech model like Whisper's encoder consumes (80 mel bands × time frames).

## M

**Mel scale / mel spectrogram** — A spectrogram whose frequency axis is warped to the **mel** scale, which spaces frequencies the way human hearing does (fine detail low, coarser high). "Spectrogram compressed into human-hearing-friendly bands." Whisper uses **80** mel bands.

**MMS-TTS (`facebook/mms-tts-eng`)** — The light **VITS**-based text-to-speech model the notebook runs locally on CPU (no speaker embeddings).

## Q

**Quantization** — Rounding each continuous amplitude sample to one of a finite number of levels (`2^bits`). Fewer bits = coarser rungs = more audible noise. The notebook demos 16→8→4→2 bits and the rising noise floor.

## R

**RMS (Root Mean Square)** — A per-frame **loudness/energy** measure: `sqrt(mean(frame²))`. High RMS = loud frame. Used with ZCR to separate voiced from unvoiced.

## S

**Sample** — One measured amplitude value at one instant. A waveform is a sequence of samples; "each dot = one sample" when you zoom in.

**Sampling rate (SR)** — How many samples per second were recorded (Hz). The notebook standardizes on **16,000 Hz (16 kHz)** — the speech-model convention. `samples = SR × seconds`.

**SNR (Signal-to-Noise Ratio)** — How much louder the true signal is than the noise, in dB. Higher = cleaner. Quantization SNR climbs ~6 dB per added bit.

**soundfile (sf)** — Library used to read/write WAV (`sf.read`/`sf.write`), chosen to avoid a torchcodec/ffmpeg decoding issue and to cache clips offline.

**Spectrum** — Energy per frequency at a point, from an FFT. The "recipe" of frequencies a sound contains.

## T

**TTS (Text-to-Speech)** — Text → speech (synthesis). The notebook uses local **MMS-TTS (VITS)** and OpenAI's `gpt-4o-mini-tts` (voice "nova") via API.

**Tone (pure tone)** — A single sine wave at one frequency. Summing several pure tones makes one more complex wave (a chord) — a single array carrying all of them.

## V

**VITS** — The neural TTS architecture behind `facebook/mms-tts-eng`. You don't need its internals — just "lightweight TTS model that runs on CPU."

**Voiced vs unvoiced** — **Voiced** sounds (vowels) are periodic: loud (high RMS) + few zero-crossings (low ZCR). **Unvoiced** sounds (hiss, "s"/"f") are noisy: quieter + many zero-crossings (high ZCR). The notebook separates them with an RMS+ZCR heuristic (real systems use **VAD**).

**VAD (Voice Activity Detection)** — The real (vs heuristic) technique for detecting where speech is present in audio. Named as the production-grade version of the RMS/ZCR demo.

## W

**Whisper** — OpenAI's open ASR model family (`tiny < base < small < medium < large-v3 < large-v3-turbo`). The notebook defaults to `whisper-base.en` (fast) and names `large-v3-turbo` as the production target. Front-end: 80-mel log-mel, 25 ms window, 10 ms hop.

**WER (Word Error Rate)** — The standard ASR accuracy metric: how many words the transcription got wrong vs the reference (lower = better).

## Z

**ZCR (Zero-Crossing Rate)** — How often the waveform crosses zero per frame (sign flips). High ZCR = noisy/high-frequency (unvoiced); low ZCR = periodic (voiced). Paired with RMS to classify frames.

[🔝 Back to top](#top)
