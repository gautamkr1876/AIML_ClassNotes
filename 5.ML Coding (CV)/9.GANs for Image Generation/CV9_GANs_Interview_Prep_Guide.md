<a id="top"></a>
# CV Notebook 9 — GANs for Image Generation (Deep Dive)

> Per-notebook companion to the master guide. For the full module + cross-cutting cheat sheet / glossary / drill, see [`../CV_Revision_Guide.md` §9](../CV_Revision_Guide.md#9-module9).

## What this notebook actually demonstrates

**Generate anime faces** from random noise. A **DCGAN** (Deep Convolutional GAN) implementation following the canonical 2015 Radford-Metz-Chintala recipe:
- **Generator** (G): `(1, 1, 128)` noise → `(64, 64, 3)` image via strided `Conv2DTranspose`.
- **Discriminator** (D): `(64, 64, 3)` image → `(1,)` probability via strided `Conv2D`.
- **Loss:** Binary cross-entropy on both sides (real vs fake).
- **Optimizer:** `Adam(lr=2e-4, β1=0.5)` (low β1 is **critical** for GANs).
- **Dataset:** Anime Face dataset, ~63,565 images at 64 × 64 × 3.

## 🪜 Mental anchors for this notebook

- **Counterfeiter (G) vs police (D).** They co-evolve. At Nash equilibrium, D outputs 0.5 for everything because G is making indistinguishable fakes.
- **Saturating vs non-saturating G loss.** Saturating: `log(1 − D(G(z)))` (vanishes when D is winning). Non-saturating: `−log D(G(z))` (always has gradient). **Use the non-saturating form.**
- **DCGAN recipe is sacred.** Strided convs, BN everywhere except G output / D input, LeakyReLU(0.2) in D, ReLU in G, tanh on G output, Adam(2e-4, β1=0.5), images normalized to `[-1, 1]`. Deviate at your peril.

## 🧠 Key cell-by-cell walkthrough

### 1. Dataset prep — normalize to [-1, 1] (to match tanh)
```python
real_ds = tf.keras.utils.image_dataset_from_directory(
    'anime_faces', image_size=(64, 64), batch_size=512, labels=None,
)
real_ds = real_ds.map(lambda x: (tf.cast(x, tf.float32) - 127.5) / 127.5)
```

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

### 5. Sample new images
```python
z = tf.random.normal((16, 1, 1, 128))
fakes = generator(z, training=False)
fakes = (fakes + 1) / 2          # rescale [-1, 1] → [0, 1] for display

fig, axs = plt.subplots(4, 4, figsize=(8, 8))
for ax, img in zip(axs.flatten(), fakes):
    ax.imshow(img.numpy()); ax.axis('off')
```

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

## ⚠️ Notebook-specific gotchas

1. **`Adam(β1=0.5)` is critical.** Default `β1=0.9` makes Adam too momentum-heavy; GANs need fast adaptation. Skip this and your GAN won't train.
2. **Normalize images to `[-1, 1]`** so they match G's `tanh` output range. `[0, 1]` (sigmoid output) also works but the convention is tanh + `[-1, 1]`.
3. **NO BatchNorm on D's input layer or G's output layer.** D needs to see the raw input distribution; G's output should be unbounded except for the final tanh.
4. **Use the non-saturating G loss.** `−log D(G(z))` (target = 1 with BCE) — strong gradient throughout. The original minimax `log(1 − D(G(z)))` vanishes when D is winning.
5. **Loss curves are misleading.** They oscillate, don't monotonically decrease. **Always inspect sample images** every few epochs.
6. **Mode collapse:** G finds a few outputs that consistently fool D. Diagnose by sampling many latents — if all images look similar, you have collapse. Fix with WGAN-GP, spectral norm, or minibatch discrimination.
7. **D too strong → G can't learn.** If D's accuracy hits ~100% quickly, lower D's LR or train G multiple steps per D step.
8. **`training=True` flag on BN at inference** would use batch statistics — bad for generating clean samples. Always call `generator(z, training=False)` at sample time.

## 🎯 Notebook quiz cells

**Q1.** Why LeakyReLU instead of ReLU in the discriminator?
→ Standard ReLU's zero slope for negative inputs blocks gradient flow → G receives no useful feedback → training stalls. LeakyReLU's 0.2x slope keeps gradients alive.

**Q2.** Why `tanh` on the generator output instead of `sigmoid`?
→ Outputs in `[-1, 1]`, matching how the real images are normalized. Sigmoid outputs `[0, 1]` — mismatched ranges hurt convergence.

**Q3.** What is mode collapse?
→ Generator finds a few "safe" outputs that consistently fool D, ignoring most of the data distribution. Generated samples look very similar to each other.

**Q4.** Why does the discriminator loss often increase over training?
→ Generator is improving, making fakes harder to classify. Higher D loss reflects a harder classification task, not a worse model.

**Q5.** Why `β1 = 0.5` for Adam in GANs?
→ Lower momentum helps the network converge faster to the moving target (since the other network keeps changing). Default `β1 = 0.9` is too smooth — Adam can't keep up.

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

## How to evaluate GANs

You cannot use loss. Standard metrics:

1. **Visual inspection** — generate a grid every few epochs; look for diversity and realism.
2. **Fréchet Inception Distance (FID)** — embed real and generated images via Inception-v3; fit Gaussians; compute Fréchet distance between them. **Lower = better.** This is the production-grade metric.
3. **Inception Score (IS)** — older metric; has documented flaws (favors a few clear classes over diverse natural images). Mostly historical.
4. **Precision / Recall** for generative models (Kynkäänniemi 2019) — precision = sample quality, recall = mode coverage. More nuanced than FID.

## End of CV journey

Notebook 9 closes the loop: from CNNs that *recognize* (1–7) to networks that *generate* (9). [Back to master guide →](../CV_Revision_Guide.md)

[🔝 Back to top](#top) | [Master guide](../CV_Revision_Guide.md)
