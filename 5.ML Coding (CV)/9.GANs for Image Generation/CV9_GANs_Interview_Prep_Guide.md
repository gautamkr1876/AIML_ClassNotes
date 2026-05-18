<a id="top"></a>
# CV Notebook 9 — GANs for Image Generation (Deep Dive)

> Per-notebook companion to the master guide. For the cross-cutting cheat sheet / glossary / drill, see [`../CV_Revision_Guide.md` §9](../CV_Revision_Guide.md#9-module9). This deep dive is **standalone** — every concept below carries its own full Concept Definition Template entry (mental model + what + why + how + where + related + code + gotcha), substituted with the notebook's actual shapes, code, and dataset (anime faces, DCGAN, 128-D latent vectors, BCE losses). GAN math is the most beginner-hostile in the CV module — every formula here is translated symbol by symbol into English. You should never need to click through to the master to understand a term.

## What this notebook actually demonstrates

- **Task: generate anime faces** from random noise. A **DCGAN** (Deep Convolutional GAN) implementation following the canonical 2015 Radford-Metz-Chintala recipe.
- **Generator (G):** input `(1, 1, 128)` noise → output `(64, 64, 3)` image via strided `Conv2DTranspose` layers, `BatchNorm`, `LeakyReLU`, and a final `tanh`.
- **Discriminator (D):** input `(64, 64, 3)` image → output `(1,)` probability via strided `Conv2D` layers, `LeakyReLU`, and a final `sigmoid`.
- **Loss:** **Binary cross-entropy** on both sides (real vs fake). G uses the **non-saturating** formulation so gradients stay alive when D is winning.
- **Optimiser:** `Adam(lr=2e-4, β1=0.5)` — the **low β1** is the DCGAN-specific tweak that makes training stable.
- **Dataset:** Anime Face dataset, ~63,565 images at `64 × 64 × 3`, normalised to `[-1, 1]` to match G's `tanh` output.

## 🪜 Mental anchors for this notebook

- **Counterfeiter (G) vs police (D).** They co-evolve. At Nash equilibrium, D outputs 0.5 for everything because G is making indistinguishable fakes.
- **Every image starts from random noise.** G takes a 128-D vector sampled from a Gaussian — there's no "input image." That's why GANs are *generative* and not *transforming*.
- **Saturating vs non-saturating G loss.** Saturating: `log(1 − D(G(z)))` (gradient vanishes when D is winning). Non-saturating: `−log D(G(z))` (always has gradient). **Use the non-saturating form** — this is non-negotiable.
- **DCGAN recipe is sacred.** Strided convs (no pooling), BN everywhere except G output and D input, LeakyReLU(0.2) in D, ReLU in G, tanh on G output, Adam(2e-4, β1=0.5), images normalised to `[-1, 1]`. Deviate at your peril.
- **Loss curves lie.** They oscillate and *should* — they're not monotonically decreasing. Inspect samples every few epochs.

[🔝 Back to top](#top)

## 📖 Concept walkthroughs

### Generative vs discriminative models

> **🪜 Mental model:** *Discriminative draws the line; generative draws the picture.* A discriminative model decides which side of a boundary your data falls on; a generative model invents new data points that look like they came from your training distribution.

**What it is.** Machine-learning models split into two big families. **Discriminative** models learn `P(y | x)` — given an input `x` (image, sentence, …), predict a label `y` (class, score, box, mask). Classifiers, detectors, and Modules 1–8 of this CV track are all discriminative. **Generative** models learn `P(x)` (or `P(x, y)`, or `P(x | y)`) — the probability distribution of the *data itself*. Once you have a generative model, you can *sample* from it to produce new examples — new images, new sentences, new molecules. GANs are an *implicit* generative model: they don't write down `P(x)` explicitly but learn to sample from it via the adversarial game.

**Why it matters.** Generative models are the entire substrate of modern AI's most visible breakthroughs — image generation (DALL·E, Midjourney, Stable Diffusion), text generation (GPT family), code generation, voice cloning. The conceptual shift is huge: a generative model doesn't *answer questions about data*, it *produces data*. Every CV interview now includes at least one question that probes whether candidates understand the discriminative-vs-generative split clearly. This notebook is the entry point — the simplest possible generative-image model (DCGAN) on the simplest possible task (unconditional anime faces).

**How it works (the three formal variants).**
- **Discriminative `P(y | x)`** — classifier, segmenter, detector. Input → label. Module 1's CNN, Module 8's Siamese.
- **Generative (unconditional) `P(x)`** — GAN's generator, VAE, diffusion. Random noise → image. **This notebook.**
- **Conditional generative `P(x | y)`** — cGAN, text-to-image, ControlNet. Class label or text prompt → image of that class.
- **Joint generative `P(x, y)`** — older Bayesian approach (Naive Bayes, GMM); fits both `P(x | y)` and `P(y)` and can be flipped via Bayes' rule to do classification.

GANs sit in the implicit-generative camp: they learn to sample from `P(x)` without ever writing the density down. You can generate; you can't directly evaluate the probability of a given image. (VAEs and diffusion models give you both, with tradeoffs.)

**Where it's used.**
- **Generative** — image synthesis (this notebook, StyleGAN, Stable Diffusion), text generation (LLMs), audio generation (WaveNet), molecule design.
- **Discriminative** — every classifier, detector, segmenter, recommender system, fraud-detection score.
- **Conditional generative** — text-to-image (DALL·E, Stable Diffusion), pix2pix, super-resolution, cGAN.

**Related terms.**
- **Implicit vs explicit generative** — implicit (GAN) lets you sample but not evaluate `P(x)`; explicit (VAE, autoregressive, flow, diffusion) gives a tractable density.
- **Sampling** — drawing a new example from the learned distribution; for GANs, this is just `G(z)` with fresh `z`.
- **Likelihood** — `P(x)` evaluated at a specific `x`; GANs *cannot* give you this directly (the famous "no likelihood" problem).
- **Mode coverage** — does the generative model produce samples from *all* parts of the data distribution? Mode collapse is a failure of mode coverage.
- **Discriminator** — confusingly, the *discriminative* half of a GAN; named because it discriminates real from fake.

```python
# Discriminative — Modules 1-8 paradigm
y_pred = classifier(image)            # P(y | x)

# Generative — this notebook
z = tf.random.normal((batch, 1, 1, 128))   # sample latent
x_fake = G(z)                              # draw from learned P(x)
```

**Gotcha.** "Generative AI" gets used loosely. Strictly, "generative" means "models a probability distribution over data" — not "uses a transformer" or "writes text." A classifier built on a transformer is still discriminative.

### Generator (G)

> **🪜 Mental model:** *Reverse CNN — noise in, image out.* Where a classifier compresses (H × W → label), the generator inflates (latent vector → H × W image).

**What it is.** The **generator** `G` is the half of a GAN that produces fake images. Its job is to learn a function `G: z → x` mapping a random noise vector `z` (sampled from a Gaussian) to a synthetic image. Once trained, `G` is the *only* network you keep — the discriminator is discarded. Architecturally, `G` is the spatial mirror of a CNN classifier: it starts with a low-resolution / many-channels representation and uses transposed convolutions (or upsampling + conv) to grow the spatial dimensions while shrinking the channel count, ending at the target image shape.

**Why it matters.** `G` is where the generative power lives — at inference, *all* you do is run `G(z)` for random `z`s. Understanding its architectural choices is essential for any GAN interview: why transposed convolutions, why BatchNorm everywhere except the output, why `tanh`, why `LeakyReLU` or `ReLU`. Each rule has a documented failure mode if you skip it, and DCGAN interviews lean heavily on these.

**How it works (this notebook's specific architecture).**
1. **Input:** noise tensor of shape `(batch, 1, 1, 128)` — `1×1` spatial, 128 channels. Sampled from `N(0, I)`.
2. **Block 1 — `Conv2DTranspose(512, kernel=4, strides=1, padding='valid')`**: expands from `1×1` to `4×4` spatial; 512 channels. Followed by `BatchNorm(momentum=0.5)` + `LeakyReLU(0.2)`.
3. **Block 2 — `Conv2DTranspose(256, kernel=4, strides=2, padding='same')`**: doubles spatial to `8×8`, halves channels to 256.
4. **Block 3 — `Conv2DTranspose(128, ...)`**: `16×16×128`.
5. **Block 4 — `Conv2DTranspose(64, ...)`**: `32×32×64`.
6. **Block 5 — `Conv2DTranspose(3, ...)`**: `64×64×3` — image-shaped. **No BatchNorm here** (DCGAN rule).
7. **Output:** `tanh` activation → pixel values in `[-1, 1]` matching the normalised real images.

Spatial size grows `1 → 4 → 8 → 16 → 32 → 64`; channel count shrinks `128 → 512 → 256 → 128 → 64 → 3`. Doubling-then-halving is the visual signature of a generator.

**Where it's used.**
- **At training:** as one half of the adversarial loop — produces fakes for D to grade.
- **At inference:** by itself — sample `z`, get an image. The discriminator is discarded.
- **In modern GANs:** the same general shape with progressive growing (Progressive GAN), style injection (StyleGAN), or self-attention (SAGAN).
- **In diffusion models:** the U-Net denoiser plays a similar role — `noise → image` — but the noise removal is iterative rather than one-shot.

**Related terms.**
- **Discriminator** — the adversarial counterpart; takes images, outputs real/fake probability.
- **Transposed convolution (`Conv2DTranspose`)** — learnable upsampling layer used throughout `G`.
- **Latent vector `z`** — the random input to `G`; lives in a latent space.
- **BatchNorm** — used in every middle block of `G` to stabilise training; **omitted from the final layer** so the output range isn't fought against by BN's affine.
- **`tanh` output** — bounds pixels to `[-1, 1]`; pair with `(x − 127.5) / 127.5` real-image normalisation.

```python
# This notebook's generator skeleton (5 blocks, channel halving)
inp = Input(shape=(1, 1, 128))
x = layers.Conv2DTranspose(512, 4, strides=1, padding='valid')(inp)
x = layers.BatchNormalization(momentum=0.5)(x); x = layers.LeakyReLU(0.2)(x)
# ... three more Conv2DTranspose+BN+LeakyReLU blocks ...
x = layers.Conv2DTranspose(3, 4, strides=2, padding='same')(x)   # NO BN here
out = layers.Activation('tanh')(x)
generator = Model(inp, out, name='generator')
```

**Gotcha.** Forgetting to omit BatchNorm on the final layer is a classic DCGAN bug — BN's affine parameters fight the `tanh` for control of the output range, producing washed-out or saturating samples.

### Discriminator (D)

> **🪜 Mental model:** *Standard CNN classifier — image in, "real or fake?" out.* Architecturally normal; the magic is in *what* it's trained to distinguish (real images vs G's fakes).

**What it is.** The **discriminator** `D` is the half of a GAN that judges images. Its job is to learn a function `D: x → [0, 1]` outputting the probability that image `x` is real (drawn from the training data) rather than fake (produced by `G`). Architecturally, `D` is a plain image classifier — strided convolutions for downsampling, LeakyReLU activations, a final dense layer with `sigmoid`. The difference from a standard classifier is the *label*: real images are labelled `1`, G's fakes are labelled `0`, and the labels change every batch as `G` improves.

**Why it matters.** `D`'s gradient is the *only* learning signal `G` ever receives. If `D` is too weak, `G` gets no useful feedback and converges to nothing. If `D` is too strong, it perfectly classifies fakes as fakes — the gradient flowing back into `G` vanishes (the saturation problem). Balancing the two networks' strengths is the central engineering challenge in GAN training and the source of most failure modes (vanishing gradient, mode collapse). At deployment, `D` is **discarded** — once `G` is trained, you don't need a judge anymore.

**How it works (this notebook's specific architecture).**
1. **Input:** image tensor of shape `(batch, 64, 64, 3)` — pixel values in `[-1, 1]`.
2. **Block 1 — `Conv2D(64, kernel=4, strides=2, padding='same')`**: downsamples to `32×32×64`. Followed by `LeakyReLU(0.2)`. **No BatchNorm** (DCGAN rule — D needs to see the raw input distribution).
3. **Block 2 — `Conv2D(128, kernel=4, strides=2)` + BN + LeakyReLU**: `16×16×128`.
4. **Block 3 — `Conv2D(256, ...)` + BN + LeakyReLU**: `8×8×256`.
5. **Block 4 — `Conv2D(512, ...)` + BN + LeakyReLU**: `4×4×512`.
6. **Flatten + `Dense(1, activation='sigmoid')`** → scalar probability per image.

Spatial size shrinks `64 → 32 → 16 → 8 → 4`; channel count grows `64 → 128 → 256 → 512`. Halving-then-doubling is the visual signature of a discriminator (and a regular CNN classifier).

**Why LeakyReLU instead of plain ReLU?** Plain ReLU's zero gradient on negative inputs blocks signal flow when `D`'s intermediate features go negative — `G` then receives no useful gradient through those paths. `LeakyReLU(0.2)` keeps a `0.2x` slope for negatives so gradient always flows.

**Where it's used.**
- **At training:** half of the adversarial loop; updated to push `D(real) → 1` and `D(fake) → 0`.
- **At inference: discarded.** This sometimes confuses beginners — yes, all that training effort produces a throwaway network.
- **In WGAN:** replaced by a "critic" that outputs an unbounded score instead of a sigmoid probability.
- **In conditional GANs:** receives the class label as additional input so the same `D` can grade per-class realism.

**Related terms.**
- **Generator** — the adversarial counterpart; you keep it at inference, discard D.
- **LeakyReLU** — the canonical D activation; keeps gradient alive for negative pre-activations.
- **Sigmoid** — final activation for vanilla / DCGAN; WGAN replaces it with a linear output.
- **Critic** — WGAN's renaming of D; gradient-Lipschitz constrained.
- **Spectral normalisation** — regularisation technique that limits D's expressivity to prevent it from overwhelming G.

```python
# This notebook's discriminator skeleton (4 strided convs, no BN on input layer)
inp = Input(shape=(64, 64, 3))
x = layers.Conv2D(64, 4, strides=2, padding='same')(inp)             # NO BN here
x = layers.LeakyReLU(0.2)(x)
# ... three more Conv2D+BN+LeakyReLU blocks ...
x = layers.Flatten()(x)
out = layers.Dense(1, activation='sigmoid')(x)
discriminator = Model(inp, out, name='discriminator')
```

**Gotcha.** If you put BatchNorm on D's *input* layer, BN's running averages will smooth away the very statistics D needs to see to distinguish real from fake. Always leave the first D layer un-normalised.

### Adversarial training loop — the minimax game in plain English (full template — the central concept of the notebook)

> **🪜 Mental model:** *Counterfeiter and detective, taking turns.* On each training step: (1) update the detective so it gets better at spotting current fakes. (2) update the counterfeiter so it gets better at fooling the (just-improved) detective. Repeat.

**What it is.** **Adversarial training** is the procedure that makes GANs work. Two networks — generator `G` and discriminator `D` — are trained alternately, each trying to defeat the other. Formally, this is a **minimax** game:

`min_G  max_D  V(D, G) = E_{x ~ p_data}[log D(x)]  +  E_{z ~ p_z}[log(1 − D(G(z)))]`

Translating every symbol into words:
- `E_{x ~ p_data}[…]` = "average over real images `x` drawn from the real-data distribution `p_data`" — i.e., over your training set.
- `E_{z ~ p_z}[…]` = "average over noise vectors `z` drawn from the noise distribution `p_z`" — typically a standard Gaussian.
- `D(x)` = the probability D assigns to image `x` being real. Range `[0, 1]`.
- `G(z)` = the fake image produced by G from noise `z`.
- `D(G(z))` = D's probability that G's fake is real.
- `log D(x)` is large (near 0) when D correctly says "real" on a real image. So `max_D` wants this term big.
- `log(1 − D(G(z)))` is large when D correctly says "fake" on a fake image (i.e., `D(G(z))` is near 0). So `max_D` wants this term big too.
- `min_G` wants the *whole thing small* — but G can only influence the second term (it doesn't control real images). To make `log(1 − D(G(z)))` small, G must drive `D(G(z))` toward 1 — i.e., fool D.

At Nash equilibrium, `D(x) = 0.5` for both real and fake images — D can't tell them apart, and G has learned to sample from `p_data`.

**Why it matters.** This co-evolution is *the* idea behind GANs. Each network's progress depends on the other's; if either gets too far ahead, training collapses (vanishing-gradient if D wins; mode collapse if G "hacks" D). Understanding the minimax structure is essential for diagnosing GAN failures and for every GAN interview question.

**How it works — the loop spelled out (matching this notebook's `train_step`).**

```
For each training batch of real images:

  D step  -----------------------------------------------------
    1. Sample noise z ~ N(0, I), shape (batch, 1, 1, 128).
    2. Generate fakes:    x_fake = G(z)
    3. Concat real + fake batches.
    4. Labels: 1 for real, 0 for fake.
    5. Compute d_loss = BCE(D(x_real), 1) + BCE(D(x_fake), 0).
    6. Backprop through D only.       ← G's weights frozen this half-step
    7. Apply gradient update to D.

  G step  -----------------------------------------------------
    8. Sample fresh noise z ~ N(0, I).
    9. Generate fakes:    x_fake = G(z)
   10. Compute g_loss = BCE(D(x_fake), 1).      ← non-saturating form
   11. Backprop through G only (gradients flow through D but only G is updated).
   12. Apply gradient update to G.

Repeat for many epochs.
```

Step 10 is the **non-saturating G loss**: G is told "your fakes should be labelled real (=1)." Equivalent to minimising `−log D(G(z))`. The original minimax formulation `min_G log(1 − D(G(z)))` has the right *direction* but a *vanishing gradient* — when `D(G(z)) ≈ 0` (D winning), `log(1 − 0)` ≈ 0, and the derivative is tiny. Flipping to `−log D(G(z))` keeps a strong gradient throughout. **This is the single most important practical tweak in GAN training.**

**Where it's used.** Inside the training loop of every GAN — DCGAN, WGAN, StyleGAN, CycleGAN, BigGAN. The alternating D-then-G structure is universal; the specifics (loss formula, number of D steps per G step, gradient penalty, etc.) vary.

**Related terms.**
- **Nash equilibrium** — the theoretical converged state (`D = 0.5` everywhere).
- **Non-saturating loss** — the practical G-loss formula used in this notebook.
- **TTUR (Two Time-Scale Update Rule)** — modern variant: D and G use different learning rates.
- **k:1 update ratio** — some implementations do k D-steps per G-step; this notebook uses 1:1.
- **Vanishing gradient** — failure mode when D wins too hard.

```python
# Simplified DCGAN train_step (matches this notebook)
def train_step(real):
    z = tf.random.normal((batch_size, 1, 1, z_dim))

    # D step
    fake = G(z)
    with tf.GradientTape() as t:
        d_real = D(real)
        d_fake = D(fake)
        d_loss = bce(tf.ones_like(d_real), d_real) + bce(tf.zeros_like(d_fake), d_fake)
    apply(t.gradient(d_loss, D.trainable_weights), D)

    # G step (non-saturating)
    with tf.GradientTape() as t:
        d_fake = D(G(z))
        g_loss = bce(tf.ones_like(d_fake), d_fake)   # G wants D to say "real"
    apply(t.gradient(g_loss, G.trainable_weights), G)
```

**Gotcha.** When D is too strong, G receives near-zero gradient even with the non-saturating loss — both `D(G(z))` and `1 − D(G(z))` are bounded near 0 or 1. Symptom: D loss → 0 quickly while G loss explodes. Fix: lower D's learning rate, add noise to D's inputs (instance noise), use spectral normalisation, or switch to Wasserstein loss.


### Latent space `z` and sampling — every image starts from noise (full template)

> **🪜 Mental model:** *Recipe dial.* The noise vector is a coordinate in an abstract space; different coordinates produce different images. Smooth movements in z produce smooth interpolations in image space.

**What it is.** The **latent vector** `z` is the random input to G. Every fake image starts as a vector of typically 100–256 numbers, each sampled independently from a standard Gaussian (`z ~ N(0, I)`). The collection of all possible `z` values is the **latent space**. G is a learned function `z → image` — once trained, *the entirety of G's image-generation ability lives in this mapping.* Different `z`s produce different images; linear interpolations in `z`-space often produce smooth morphs in image space (e.g., interpolating between two `z`s slowly morphs one anime face into another).

**Why it matters.** Beginners often expect a GAN to take an "input image" — but G has **no input image, ever.** It only ever sees random noise. This is the single most surprising thing about generative models for first-timers. The implication: the model has memorised the data distribution in such a way that random points in noise space deterministically land on plausible images. The latent space is where all the "magic" happens — interpolations, attribute-direction arithmetic (`z_smiling = z_neutral + smiling_direction`), and StyleGAN-style attribute editing all happen by manipulating `z` (or its richer cousin `w`).

**How it works (this notebook's specifics).**
1. `z_dim = 128` — each fake image is generated from a 128-D vector.
2. Shape is `(batch, 1, 1, 128)` — the leading `(1, 1)` is there because G's first layer is a `Conv2DTranspose(filters=512, kernel=4, strides=1, padding='valid')` that expects a `(1, 1, 128)` spatial input. (Other implementations use a flat `(batch, 128)` vector + a Dense layer; same idea, different reshape.)
3. Sampling: `z = tf.random.normal((batch_size, 1, 1, 128))` — independent draws every step. No memory across batches.
4. Forward pass: `z` → `Conv2DTranspose` → `(4, 4, 512)` → … → `(64, 64, 3)`.
5. At inference, you re-sample fresh `z`s to generate new images — `generator(tf.random.normal((16, 1, 1, 128)), training=False)` yields 16 fresh anime faces.

**Where it's used.** Image generation (sample `z`, get a new image). Latent-space arithmetic (interpolate between two `z`s for smooth morphs). StyleGAN's attribute editing (move `z` in specific directions). Generative super-resolution (`z` represents the missing high-frequency content). Anomaly detection (any image G can't reproduce from any `z` is "out of distribution").

**Related terms.**
- **Latent space** — the space `z` lives in. Usually `R^128` or `R^256` for DCGAN; richer spaces for StyleGAN (`W`, `W+`).
- **Embedding** — Module 8's notion; similar shape but learned for *similarity*, not generation.
- **Truncation trick** — sample `z` from a *truncated* Gaussian (clipped to `[-c, c]`) to bias toward "average" images; common in StyleGAN.
- **Disentangled latent space** — what StyleGAN aims for: each dimension of `z` (or `w`) controls one semantic attribute.
- **Reparameterisation trick** — VAE's variant; relevant for understanding why `z ~ N(0, I)` is the standard prior.

```python
# Sample a batch of 16 fakes from random noise
z = tf.random.normal((16, 1, 1, 128))      # 16 fresh latent vectors
fakes = generator(z, training=False)        # G is deterministic given z
fakes = (fakes + 1) / 2                     # rescale [-1, 1] → [0, 1] for display
```

**Gotcha.** Use `training=False` when sampling at inference. Otherwise `BatchNorm` would use the current batch's statistics instead of its trained running averages — your samples will look noisy or distorted compared to the same `z` evaluated with `training=False`. (This is the BN-mode trap in many GAN tutorials.)


### Loss functions for D and G — the BCE math, spelled out (full template — the math beginners hate)

> **🪜 Mental model:** *Two binary classifiers, opposite goals.* D is trained to call real images "1" and fakes "0"; G is trained to make D call its fakes "1". Both use the same binary-cross-entropy formula — the difference is in the labels they're trained against.

**What it is.** Both losses in DCGAN are **binary cross-entropy** (BCE). The BCE between a predicted probability `p ∈ [0, 1]` and a binary label `y ∈ {0, 1}` is:

`BCE(y, p) = −[ y · log(p)  +  (1 − y) · log(1 − p) ]`

Translating word by word:
- When `y = 1` (label says "real"), the formula reduces to `−log p`. This is small (good) when `p ≈ 1` and large (bad) when `p ≈ 0`. The classifier is rewarded for high confidence on real.
- When `y = 0` (label says "fake"), the formula reduces to `−log(1 − p)`. This is small when `p ≈ 0` (low probability of real = correct for fake). Rewarded for low confidence on fake.

Both halves are non-negative and minimised at the right answer. The full BCE just averages the right one over the batch.

**The two losses in this notebook.**

**1. Discriminator loss (`d_loss`):**

```
d_loss = BCE(label=1, D(x_real))  +  BCE(label=0, D(x_fake))
       = −log D(x_real)            −  log(1 − D(x_fake))
```

In words: *"Punish D when it gives a low score to a real image; punish D when it gives a high score to a fake."* D minimises this — i.e., it pushes `D(x_real) → 1` and `D(x_fake) → 0`.

**2. Generator loss (`g_loss`) — non-saturating form:**

```
g_loss = BCE(label=1, D(x_fake))    ← Note: label is 1 (LIE about the fake)
       = −log D(x_fake)
```

In words: *"Punish G when D gives a low score to G's fakes."* G minimises this — i.e., it pushes `D(x_fake) → 1`. **Note the deliberate "lie":** when computing G's loss, we use label = 1 *for fake images*, because we want G to be rewarded for fooling D into thinking fakes are real.

**Why it matters.** Two reasons beginners need this spelled out:
1. **The `label=1` for fakes in G's loss looks like a bug.** It isn't — it's the trick that makes the minimax direction reverse for G. G doesn't want to be honest about fakes being fake; G wants D to call them real.
2. **The non-saturating form is *not* the original minimax G-loss.** The original was `log(1 − D(G(z)))`, which vanishes when `D(G(z)) ≈ 0` (D winning). The fix — replace `log(1 − D(G(z)))` with `−log D(G(z))` — is mathematically equivalent in spirit (both push `D(G(z)) → 1`) but has *strong gradient* even when D is winning. Goodfellow's original paper used the saturating form theoretically and the non-saturating form practically. Every modern implementation uses the non-saturating one.

**How it works — one training step, fully traced.**

```
z = sample N(0, I)                # shape (batch, 1, 1, 128)
x_fake = G(z)                     # shape (batch, 64, 64, 3)

# --- D step ---
d_real = D(x_real)                # shape (batch, 1) — D's "is this real?" probs
d_fake = D(x_fake)                # shape (batch, 1)
d_loss = BCE(ones,  d_real) + BCE(zeros, d_fake)
backward, update D

# --- G step ---
z' = sample N(0, I)               # fresh noise (could reuse z, but fresh is cleaner)
x_fake' = G(z')
d_fake' = D(x_fake')              # D's score for the new fakes
g_loss  = BCE(ones, d_fake')      # ← "lie": pretend fakes are real
backward, update G  only
```

**Where it's used.** Every vanilla / DCGAN-style GAN. Wasserstein GAN replaces BCE with a different distance (the Wasserstein / Earth Mover's distance) — different math, same alternating-step structure.

**Related terms.**
- **Binary cross-entropy** — the underlying loss function.
- **Saturating G loss (`log(1 − D(G(z)))`)** — the original Goodfellow form; suffers from vanishing gradient.
- **Non-saturating G loss (`−log D(G(z))`)** — the practical fix; what this notebook uses.
- **Wasserstein loss** — alternative GAN loss that avoids the vanishing-gradient issue by design.
- **Label smoothing** — a regularisation trick: train D on `0.9` for real instead of `1.0` to discourage over-confidence.

```python
# In this notebook's train_step, BCE is computed via keras.losses.BinaryCrossentropy()
loss_fn = keras.losses.BinaryCrossentropy()

# D's two-term loss
d_loss = loss_fn(tf.concat([tf.ones(bs, 1), tf.zeros(bs, 1)], axis=0),
                 D(tf.concat([real, fake], axis=0)))

# G's one-term loss — labels are all ones (the deliberate lie)
g_loss = loss_fn(tf.ones((bs, 1)), D(G(z)))
```

**Gotcha.** D's loss can stay low while G's loss blows up — that means D is "winning" too easily and G is getting no useful gradient. Symptom: D loss near 0; G loss climbing each epoch; samples don't improve. Fix: lower D's learning rate, add noise to D inputs, or use the WGAN-GP loss instead.


### Mode collapse

> **🪜 Mental model:** *G finds a cheat code.* Instead of learning to cover the entire data distribution, the generator collapses onto a handful of "safe" outputs that consistently fool D — high realism, near-zero diversity.

**What it is.** **Mode collapse** is a GAN failure mode where the generator learns to produce only a small subset of the data distribution's "modes" (clusters of similar real examples) — sometimes just a single image — while ignoring the rest. The samples it does produce can look perfectly realistic; what's missing is *diversity*. Two flavours: **complete collapse** (all `z`s map to nearly the same image) and **partial / mode-dropping collapse** (G covers a few modes but consistently misses others — e.g., generates only frontal anime faces, never profile views).

**Why it matters.** Mode collapse is the single most common, most hated GAN failure. Loss curves typically look *normal* during collapse — D's loss doesn't spike, G's loss may even decrease — so loss-watching gives a false sense of "training is fine." The damage shows up only when you sample many `z`s and notice the samples look alike. Every GAN interview will ask about it because diagnosing it correctly is the entry-level skill for working with generative models.

**How it works (why it happens).** The mathematical pathology: the GAN objective only requires G's distribution to *overlap* with the data distribution, not to *match* it. If G finds a small region of image space that D currently can't distinguish from real, the easiest gradient path is to keep generating things from that region. D then *would* learn to call those out — but in the alternating game, by the time D catches up, G has often already shifted to another safe region. The two networks chase each other around a few modes instead of converging to the full distribution.

**How this notebook would detect it.** At sample time the notebook generates 16 images from 16 fresh `z`s:
```python
z = tf.random.normal((16, 1, 1, 128))
fakes = generator(z, training=False)
```
- **Visual check** — if all 16 outputs look near-identical despite 16 different `z`s, you have collapse. Eyeball a 10×10 grid for a stronger signal.
- **Pairwise distance check** — compute `mean(pairwise_L2(fakes))`; an abnormally low value (compared to real images) signals tight clustering = collapse.
- **FID / Inception Score on a large sample (e.g., 50K)** — collapsed G has a much higher FID than its visual quality would suggest, precisely because the generated distribution is too narrow.

**Where it's used (or rather, where it shows up).**
- Vanilla / DCGAN training on small or low-diversity datasets.
- Any GAN trained too long without monitoring sample diversity.
- *Less* common in modern WGAN-GP, StyleGAN, and BigGAN (which have architectural fixes).
- Not unique to images — text GANs, music GANs, and conditional GANs can all collapse.

**Related terms.**
- **Mode coverage / mode dropping** — the property mode collapse violates; ideally G covers all modes of the data distribution.
- **Nash equilibrium** — the theoretical converged state GANs aim for; mode collapse is a local equilibrium that traps the game.
- **FID (Fréchet Inception Distance)** — the metric that *does* detect collapse (loss doesn't).
- **WGAN-GP, spectral normalisation, minibatch discrimination** — the standard architectural fixes.
- **PacGAN** — explicitly trains D to look at *batches* of samples so it can detect collapse directly.

```python
# Cheap collapse diagnostic
fakes = generator(tf.random.normal((256, 1, 1, 128)), training=False)
fakes_flat = tf.reshape(fakes, (256, -1))
pairwise_d = tf.reduce_mean(tf.norm(fakes_flat[:, None, :] - fakes_flat[None, :, :], axis=-1))
# Compare against the same statistic computed on a batch of real images.
# If fakes' pairwise distance is much smaller than reals' → collapse.
```

**Gotcha.** *Don't trust loss curves.* A collapsed GAN can have completely normal-looking `d_loss` and `g_loss` plots. Always inspect sample diversity directly — many fresh `z`s → many distinct outputs.

### DCGAN (Deep Convolutional GAN)

> **🪜 Mental model:** *The cookbook for stable GAN training.* A specific set of architectural and hyperparameter rules that turned GAN training from "almost never works" into a reproducible recipe — the first reliably-trainable convolutional GAN.

**What it is.** **DCGAN** (Radford, Metz & Chintala, 2016) is the seminal paper that prescribed a concrete recipe for convolutional GANs. Before DCGAN, GAN training was unstable and brittle; after DCGAN, a clear set of architectural rules made it reproducible on CIFAR-10, LSUN, and faces datasets. The "recipe" is a list of about ten rules covering layer types, activations, normalisations, optimiser settings, and input scaling. Modern GANs (WGAN, StyleGAN, etc.) build on top of DCGAN's baseline; understanding DCGAN is the prerequisite for understanding any subsequent GAN.

**Why it matters.** Every GAN training run starts (implicitly or explicitly) from the DCGAN recipe. Interview questions on GANs lean heavily on "name the DCGAN rules and why each one matters" — it's the GAN equivalent of "name the standard CNN training tricks." Knowing *why* each rule exists (and what happens when you break it) separates candidates who've trained a GAN from those who've only read about them. **This notebook is essentially a faithful DCGAN implementation on anime faces.**

**How it works (every DCGAN rule, mapped to this notebook's code).**

| DCGAN rule | Why it exists | This notebook's implementation |
|---|---|---|
| Strided convs for downsampling in D (no pooling) | Pooling throws away gradient-friendly information; strided convs let D learn its own downsampling | `Conv2D(stride=2)` in every D block |
| Transposed convs for upsampling in G (no upsampling) | Learnable upsampling — G can shape its own resolution-increase | `Conv2DTranspose(stride=2)` in every G block |
| BatchNorm in G and D **except** G's output and D's input | G output BN would fight `tanh` for the output range; D input BN would smooth away the very statistics needed to distinguish real from fake | `BatchNormalization(momentum=0.5)` in all middle blocks; absent in D's first block and G's final block |
| LeakyReLU(0.2) in D | Plain ReLU's zero gradient on negatives blocks G's learning signal | `LeakyReLU(0.2)` after every D conv |
| ReLU in G (notebook uses LeakyReLU — deliberate deviation) | Sparse activations help G shape clean features | `LeakyReLU(0.2)` after every G conv (this notebook's tweak) |
| tanh on G output | Symmetric output range matches `[-1, 1]` normalised inputs; well-conditioned gradient for both bright and dark pixels | `Activation('tanh')` after G's final conv |
| Adam `lr=2e-4, β1=0.5` | Lower β1 (default 0.9 → 0.5) lets Adam adapt to the non-stationary GAN loss surface | `Adam(2e-4, β1=0.5)` for D; `Adam(1.5e-4, β1=0.5)` for G (slight TTUR) |
| Normalise images to `[-1, 1]` | Matches G's tanh output range exactly | `(x − 127.5) / 127.5` mapping |
| BN momentum `0.5` (lower than default 0.99) | Faster adaptation to the moving target of the adversarial game | `BatchNormalization(momentum=0.5)` throughout |

**Where it's used.**
- **As a baseline** for any new GAN paper — does the new idea beat DCGAN on CIFAR-10?
- **As a teaching architecture** — this notebook; almost every tutorial GAN.
- **As the parent** of modern variants — WGAN, StyleGAN, BigGAN all inherit DCGAN's conv-strided + BN structure.

**Related terms.**
- **WGAN / WGAN-GP** — replaces BCE with Wasserstein distance; more stable, fewer mode-collapse issues.
- **StyleGAN** — DCGAN's high-resolution descendant; adds style mixing + AdaIN.
- **cGAN** — DCGAN + label conditioning on both G and D.
- **Progressive GAN** — DCGAN with progressive resolution growth (start at 4×4, double to target).
- **Spectral normalisation** — a regularisation trick added on top of DCGAN to further stabilise training.

```python
# DCGAN's β1=0.5 is critical — without it, the GAN won't train
g_opt = keras.optimizers.Adam(learning_rate=1.5e-4, beta_1=0.5)
d_opt = keras.optimizers.Adam(learning_rate=2e-4,   beta_1=0.5)
```

**Gotcha.** Skipping any single DCGAN rule (especially `β1=0.5` or the BatchNorm omissions) often produces a GAN that *seems* to be training (losses look reasonable) but generates noise. Treat the recipe as non-negotiable; deviate only after you have a baseline.

[🔝 Back to top](#top)

## 🧠 Key cell-by-cell walkthrough

### 1. Dataset prep — normalize to [-1, 1] (to match tanh)
```python
real_ds = tf.keras.utils.image_dataset_from_directory(
    'anime_faces', image_size=(64, 64), batch_size=512, labels=None,
)
real_ds = real_ds.map(lambda x: (tf.cast(x, tf.float32) - 127.5) / 127.5)
```
Maps `[0, 255]` → `[-1, 1]` so the real images live in the same range as G's tanh output. Skipping this step breaks training silently.

### 2. Generator (noise → image)
```python
def make_generator(noise_shape=(1, 1, 128)):
    inp = Input(shape=noise_shape)
    x = layers.Conv2DTranspose(512, 4, strides=1, padding='valid')(inp)  # 4×4
    x = layers.BatchNormalization(momentum=0.5)(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)     # 8×8
    x = layers.BatchNormalization(momentum=0.5)(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(x)     # 16×16
    x = layers.BatchNormalization(momentum=0.5)(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(x)      # 32×32
    x = layers.BatchNormalization(momentum=0.5)(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2DTranspose(3, 4, strides=2, padding='same')(x)       # 64×64
    out = layers.Activation('tanh')(x)
    return Model(inp, out, name='generator')
```
Spatial size grows `1 → 4 → 8 → 16 → 32 → 64`; channel count shrinks `128 → 512 → 256 → 128 → 64 → 3`. Final `tanh` clamps to `[-1, 1]`. No BN on the final layer.

### 3. Discriminator (image → probability)
```python
def make_discriminator(img_shape=(64, 64, 3)):
    inp = Input(shape=img_shape)
    x = layers.Conv2D(64, 4, strides=2, padding='same')(inp)            # 32×32; NO BN here
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(128, 4, strides=2, padding='same')(x)             # 16×16
    x = layers.BatchNormalization(momentum=0.5)(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(256, 4, strides=2, padding='same')(x)             # 8×8
    x = layers.BatchNormalization(momentum=0.5)(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(512, 4, strides=2, padding='same')(x)             # 4×4
    x = layers.BatchNormalization(momentum=0.5)(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Flatten()(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return Model(inp, out, name='discriminator')
```
Mirror image of G: spatial size shrinks, channels grow. **No BN on the first layer** — D needs to see the raw input distribution.

### 4. GAN training loop (custom Keras model)
```python
class GAN(keras.Model):
    def __init__(self, G, D, z_dim=128):
        super().__init__()
        self.G, self.D, self.z_dim = G, D, z_dim

    def compile(self, g_opt, d_opt, loss_fn):
        super().compile()
        self.g_opt, self.d_opt, self.loss_fn = g_opt, d_opt, loss_fn

    def train_step(self, real):
        bs = tf.shape(real)[0]
        z  = tf.random.normal((bs, 1, 1, self.z_dim))

        # ── Train Discriminator ──
        fake = self.G(z)
        combined = tf.concat([real, fake], axis=0)
        labels   = tf.concat([tf.ones((bs, 1)), tf.zeros((bs, 1))], axis=0)
        with tf.GradientTape() as t:
            d_loss = self.loss_fn(labels, self.D(combined))
        d_grads = t.gradient(d_loss, self.D.trainable_weights)
        self.d_opt.apply_gradients(zip(d_grads, self.D.trainable_weights))

        # ── Train Generator (try to make D output 1 for fakes) ──
        with tf.GradientTape() as t:
            g_loss = self.loss_fn(tf.ones((bs, 1)), self.D(self.G(z)))
        g_grads = t.gradient(g_loss, self.G.trainable_weights)
        self.g_opt.apply_gradients(zip(g_grads, self.G.trainable_weights))
        return {'d_loss': d_loss, 'g_loss': g_loss}

gan = GAN(generator, discriminator, z_dim=128)
gan.compile(
    g_opt=keras.optimizers.Adam(1.5e-4, beta_1=0.5),
    d_opt=keras.optimizers.Adam(2e-4,  beta_1=0.5),
    loss_fn=keras.losses.BinaryCrossentropy(),
)
gan.fit(real_ds, epochs=50)
```
Plain English: every batch, train D once on real + fake (labelled correctly), then train G once on fake (labelled as 1 — the "lie"). Two separate `GradientTape`s isolate the two updates so D's gradient doesn't accidentally flow into G's update.

### 5. Sample new images
```python
z = tf.random.normal((16, 1, 1, 128))
fakes = generator(z, training=False)
fakes = (fakes + 1) / 2          # rescale [-1, 1] → [0, 1] for display

fig, axs = plt.subplots(4, 4, figsize=(8, 8))
for ax, img in zip(axs.flatten(), fakes):
    ax.imshow(img.numpy()); ax.axis('off')
```
`training=False` is essential — see the latent-space gotcha above.

[🔝 Back to top](#top)

## ⚙️ APIs introduced (specific to this notebook)

| Call | Notes |
|---|---|
| `layers.Conv2DTranspose(filters, kernel, strides, padding)` | Learnable upsample for the generator |
| `layers.LeakyReLU(alpha=0.2)` | Keeps gradient alive for negatives; essential in D |
| `layers.BatchNormalization(momentum=0.5)` | **Lower momentum** than default 0.99 for GAN stability |
| `keras.optimizers.Adam(lr=2e-4, beta_1=0.5)` | **Low β1** is the DCGAN secret sauce |
| `tf.GradientTape()` | Manual gradients — required because G and D have separate updates |
| `tf.concat([real, fake], 0)` | Combined batch for D training |
| `tf.random.normal((bs, 1, 1, z_dim))` | Sample latent noise |
| BCE loss `keras.losses.BinaryCrossentropy()` | The classic GAN loss |

### Modern variants to mention
| Variant | Twist | Use when |
|---|---|---|
| **DCGAN** (this notebook) | Strided convs + BN + LeakyReLU | Default starting point |
| **WGAN / WGAN-GP** | Wasserstein loss + gradient penalty | Mode collapse, training instability |
| **cGAN** | Pass class label to both G and D | Need conditional generation |
| **StyleGAN(2/3)** | Style mixing + AdaIN | High-res, controllable faces |
| **CycleGAN** | Two G's + two D's + cycle consistency | Unpaired image translation |
| **SRGAN** | Generator does super-resolution | Upscale low-res images |

[🔝 Back to top](#top)

## ⚠️ Notebook-specific gotchas

1. **`Adam(β1=0.5)` is critical.** Default `β1=0.9` makes Adam too momentum-heavy; GANs need fast adaptation. Skip this and your GAN won't train.
2. **Normalize images to `[-1, 1]`** so they match G's `tanh` output range. `[0, 1]` (sigmoid output) also works but the convention is tanh + `[-1, 1]`.
3. **NO BatchNorm on D's input layer or G's output layer.** D needs to see the raw input distribution; G's output should be unbounded except for the final tanh.
4. **Use the non-saturating G loss.** `−log D(G(z))` (target = 1 with BCE) — strong gradient throughout. The original minimax `log(1 − D(G(z)))` vanishes when D is winning.
5. **Loss curves are misleading.** They oscillate, don't monotonically decrease. **Always inspect sample images** every few epochs.
6. **Mode collapse:** G finds a few outputs that consistently fool D. Diagnose by sampling many latents — if all images look similar, you have collapse. Fix with WGAN-GP, spectral norm, or minibatch discrimination.
7. **D too strong → G can't learn.** If D's accuracy hits ~100% quickly, lower D's LR or train G multiple steps per D step.
8. **`training=True` flag on BN at inference** would use batch statistics — bad for generating clean samples. Always call `generator(z, training=False)` at sample time.
9. **Don't reuse the same `z` across D and G steps unintentionally.** Sampling fresh `z` for the G step (or at least re-running G on a fresh batch) prevents D from memorising the same fakes that G is training on.

[🔝 Back to top](#top)

## 🎯 Notebook quiz cells

**Q1.** Why LeakyReLU instead of ReLU in the discriminator? *(adapted from `chiphuyen/ml-interviews-book`, generative chapter)*
→ Standard ReLU's zero slope for negative inputs blocks gradient flow → G receives no useful feedback when D's intermediate features go negative → training stalls. LeakyReLU's `0.2x` slope keeps gradients alive for negatives, ensuring D's signal always reaches G.

**Q2.** Why `tanh` on the generator output instead of `sigmoid`?
→ Outputs in `[-1, 1]`, matching how the real images are normalized. Sigmoid outputs `[0, 1]` — mismatched ranges hurt convergence. The deeper reason: tanh is symmetric around 0, which gives well-conditioned gradients for both bright and dark pixels.

**Q3.** What is mode collapse? How would you detect it in this notebook? *(adapted from `alexeygrigorev/data-science-interviews`, GAN section)*
→ Generator finds a few "safe" outputs that consistently fool D, ignoring most of the data distribution. Generated samples look very similar to each other. Detect it by sampling many fresh `z`s and inspecting the diversity of outputs — *not* by looking at the loss curve, which can look normal during collapse.

**Q4.** Why does the discriminator loss often increase over training?
→ Generator is improving, making fakes harder to classify. Higher D loss reflects a harder classification task, not a worse model. At ideal Nash equilibrium, D loss should plateau near `log(2) ≈ 0.69` (D outputs 0.5 for everything).

**Q5.** Why `β1 = 0.5` for Adam in GANs? *(common FAANG generative-models question)*
→ Lower momentum helps the network converge faster to the moving target (since the other network keeps changing). Default `β1 = 0.9` is too smooth — Adam can't keep up with the non-stationary loss surface. Radford et al. (DCGAN, 2016) showed empirically that `β1 = 0.5` stabilises training across many datasets.

**Q6.** Why does the original minimax G loss `log(1 − D(G(z)))` vanish, and what's the fix? *(adapted from `chiphuyen/ml-interviews-book`)*
→ When D is winning, `D(G(z)) ≈ 0`, so `log(1 − D(G(z))) ≈ log(1) = 0`, and its derivative with respect to `G(z)` is also near 0. G gets no gradient *exactly when it needs it most*. Fix: switch G's loss to the non-saturating form `−log D(G(z))` — equivalent push direction (`D(G(z)) → 1`) but with strong gradient even when D is winning.

**Q7.** Why use separate `GradientTape`s for D and G updates?
→ Each tape records ops only for that update; computing D's gradient with G's tape would unnecessarily track G's ops, and vice versa. More importantly, applying gradients within the tape's scope guarantees you update only the intended parameters — D's gradient never accidentally bumps G's weights.

**Q8.** Why is FID a better metric than visual inspection for GAN evaluation? *(common interview)*
→ Visual inspection is subjective and doesn't catch mode collapse on a single image-grid. FID (Fréchet Inception Distance) embeds real and generated images via Inception-v3, fits Gaussians, and computes the Fréchet distance between them — measuring distributional similarity. Catches mode collapse because a collapsed G's embedding distribution is much tighter than the real one.

**Q9.** What's the difference between a generative and a discriminative model — and which family does this notebook's GAN belong to? *(adapted from `kojino/120-Data-Science-Interview-Questions`, Q on generative vs discriminative)*
→ Discriminative models learn `P(y | x)` — given an input, predict a label (every classifier from Modules 1–8). Generative models learn `P(x)` (or `P(x | y)`) — the distribution of the data itself, from which you can sample new examples. GANs are **implicit generative** — they sample from a learned `P(x)` without ever writing down the density. This notebook's anime-face GAN is unconditional generative: input is pure noise, output is a plausible image, no class label anywhere.

**Q10.** What does the latent vector `z` actually represent in a trained GAN, and what would interpolating between two `z`s do? *(adapted from `chiphuyen/ml-interviews-book`, generative chapter)*
→ `z` is a coordinate in an abstract space (`R^128` here) that the trained generator maps deterministically to images. Each `z` corresponds to one specific generated image; *all* of G's image-generation ability lives in this `z → image` mapping. Linear interpolation between two `z`s (e.g., `z_t = (1-t)·z_a + t·z_b` for `t ∈ [0, 1]`) typically produces smooth morphs between the two generated images — face A slowly turning into face B — which is *the* canonical demo of latent-space structure. In StyleGAN, specific directions in `z`-space correspond to semantic edits (age, smile, hair colour).

**Q11.** In one sentence each, name three DCGAN rules and why each one matters. *(common FAANG GAN interview)*
→ (1) `Adam(β1=0.5)` — default `β1=0.9` is too smooth for the non-stationary adversarial loss; lower β1 keeps Adam responsive. (2) No BatchNorm on G's output or D's input — BN would smooth away the very distribution G is trying to learn and D is trying to detect. (3) `tanh` on G with `[-1, 1]`-normalised real images — matched output ranges give well-conditioned gradients on both bright and dark pixels.

[🔝 Back to top](#top)

## 🪞 Extra ladder — diagnosing GAN failures

**Basic** — **loss explosion** (one loss → infinity). Lower the LR; check for `NaN` in inputs.

**Intermediate** — **mode collapse**. Sample many `z`s — if outputs are nearly identical, collapse. Try:
- WGAN-GP loss (smoother gradients)
- Spectral normalization on D
- Minibatch discrimination (D sees batch statistics)

**Advanced** — **non-convergence** (losses oscillate forever, samples don't improve). Try:
- Two Time-Scale Update Rule (TTUR) — different LRs for G and D
- Progressive growing (start at 4×4, grow to target resolution)
- Self-attention (SAGAN) or transformer-based G
- Bigger model + more data

[🔝 Back to top](#top)

## How to evaluate GANs

You cannot use loss. Standard metrics:

1. **Visual inspection** — generate a grid every few epochs; look for diversity and realism.
2. **Fréchet Inception Distance (FID)** — embed real and generated images via Inception-v3; fit Gaussians; compute Fréchet distance between them. **Lower = better.** This is the production-grade metric.
3. **Inception Score (IS)** — older metric; has documented flaws (favors a few clear classes over diverse natural images). Mostly historical.
4. **Precision / Recall** for generative models (Kynkäänniemi 2019) — precision = sample quality, recall = mode coverage. More nuanced than FID.

[🔝 Back to top](#top)

## End of CV journey

Notebook 9 closes the loop: from CNNs that *recognize* (1–7) to networks that *generate* (9). [Back to master guide →](../CV_Revision_Guide.md)

[🔝 Back to top](#top) | [Master guide](../CV_Revision_Guide.md)
