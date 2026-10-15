# SER Project — Phase 2 Explanation
**Features & Data · Week 2** — companion to `SER_Phase2_Features_and_Data.ipynb`

Phase 1 taught you what an audio signal is, what the FFT does, and how a spectrogram shows frequency over time. Phase 2 uses all of that to answer one question: **how do we turn a voice recording into numbers a model can learn emotion from?**

---

## 1. The big picture

```
.wav file → load → trim silence → fix to 3 s → features → label → dataset → split → normalise → save
```

| Output | Shape | Used by |
|---|---|---|
| Sequence tensor | `Samples × Features × Time` (e.g. N × 40 × 130) | CNN, CNN-LSTM, Attention (Phase 4) |
| Flat vector | `Samples × Features` (e.g. N × 80) | Logistic Regression, SVM, Random Forest (Phase 3) |

Why two shapes? Classical ML models need one fixed-length row per sample, so we summarise over time (mean, std). Deep models can read the time axis directly, so we keep it.

---

## 2. Day-by-day walkthrough

### Day 1 — MFCC from scratch
MFCC (Mel-Frequency Cepstral Coefficients) is the most important speech feature. The notebook implements it in NumPy, then checks it against `librosa.feature.mfcc` (difference ≈ 3e-5, i.e. identical).

The five steps:

1. **Framing + Hann window.** Speech is non-stationary, so we cut it into short overlapping frames (2048 samples, hop 512 at 22.05 kHz ≈ 93 ms window, 23 ms step). The Hann window tapers frame edges to avoid spectral leakage.
2. **FFT → power spectrum.** Each frame becomes `|FFT|²`: how much energy sits at each frequency.
3. **Mel filter bank.** Triangular filters spaced evenly on the *mel scale*, `mel = 2595·log10(1 + f/700)`. Humans resolve low frequencies finely and high frequencies coarsely; the mel scale copies that. Filters are area-normalised so each has equal energy.
4. **Log (dB).** Perceived loudness is logarithmic, and log compresses the huge dynamic range of raw power.
5. **DCT.** Neighbouring mel bands are strongly correlated; the DCT decorrelates them. Keeping the first 40 coefficients keeps the **spectral envelope** (shape of the vocal tract) and discards pitch-harmonic fine detail.

Reading an MFCC matrix `(40, frames)`: rows = *what the sound's shape is*, columns = *when*. Coefficient 0 ≈ overall energy; low-index coefficients = broad envelope; high-index = fine detail/noise.

### Day 2 — Mel scale, filter bank, Mel spectrogram
* The Hz→Mel plot shows the compression of high frequencies; the filter-bank plot shows narrow filters at low Hz and wide ones at high Hz.
* **Mel spectrogram = steps 1–4 only** (no DCT). It keeps all 64 bands, so it looks like an image, which is what a **CNN** is good at.
* **MFCC = Mel spectrogram + DCT.** Smaller and decorrelated, which suits classical ML and sequence models.
* The 8-emotion grid lets you *see* that emotions differ: angry/fear/surprise show more energy and faster modulation; sad/calm are flatter and slower.

### Day 3 — Delta MFCC and Chroma
* **Delta** = rate of change of each coefficient: `d_t = Σ n·(c[t+n] − c[t−n]) / (2·Σ n²)`. It is a local linear-regression slope. Emotion is largely about *dynamics* — angry speech changes fast, sad speech slowly — and plain MFCC only describes each frame statically. The notebook re-implements delta by hand and matches Librosa.
* **Delta-delta** = delta of delta (acceleration).
* **Chroma** folds all frequencies into 12 pitch classes (C, C#, … B), capturing pitch/harmonic content and intonation.

| Feature | Rows | Captures |
|---|---|---|
| MFCC | 40 | spectral envelope |
| Delta | 40 | speed of change |
| Delta-Delta | 40 | acceleration |
| Mel spectrogram | 64 | full band energy (CNN input) |
| Chroma | 12 | pitch classes / harmonics |

### Day 4 — Dataset table
* **RAVDESS** (primary): 24 actors, 8 emotions. File name encodes everything: `03-01-06-01-02-01-12.wav` → `modality-vocal-emotion-intensity-statement-repetition-actor`; emotion code 06 = fear, actor 12.
* **TESS** (secondary): 2 actresses (OAF, YAF), 7 emotions (no *calm*; `ps` = pleasant surprise → mapped to *surprise*).
* Result: one `DataFrame` with `path, dataset, emotion, label, speaker_id, duration`.
* Check class balance. RAVDESS *neutral* has half as many clips as the others and *calm* exists only in RAVDESS. That is why Phase 3 should report **F1 and the confusion matrix**, not accuracy alone.

### Day 5 — Extraction pipeline
Per file: `load → trim silence (top_db=30) → pad/crop to 3 s → peak-normalise → MFCC, Δ, ΔΔ, Mel(dB), Chroma`.
* Fixed length matters because models need equal-shaped inputs: 3 s at hop 512 = **130 frames**.
* Extraction runs in parallel with `joblib`.
* Flat vectors: `X_flat_mean40` (N × 40, the roadmap's example) and `X_flat` (N × 80, mean + std over time; std adds information about variability).

### Day 6 — Quality, split, normalisation
* **Sanity checks:** no NaN/Inf, sensible ranges, and a "mean MFCC per emotion" heatmap to confirm the features really differ by class.
* **Speaker-independent split** (`GroupShuffleSplit` on speaker): if one actor appears in both train and test, the model can memorise the *voice* instead of the *emotion* and your accuracy is inflated. Assertions in the notebook guarantee zero speaker overlap.
* **Normalisation:** per-coefficient z-score using **training statistics only**. Using val/test statistics would leak information. Test data will not be exactly mean 0 / std 1, and that is correct.

### Day 7 — Combinations, saving, smoke test
Feature combinations prepared for Phase 3 feature selection (stacked along the feature axis):

| Name | Composition | Features |
|---|---|---|
| `mfcc` | MFCC | 40 |
| `mfcc+delta` | MFCC + Δ | 80 |
| `mfcc+mel` | MFCC + Mel | 104 |
| `mfcc+chroma` | MFCC + Chroma | 52 |
| `all` | MFCC + Δ + ΔΔ + Mel + Chroma | 196 |

Saved to `features/`:
* `ser_features.npz`: all feature tensors, flat vectors, labels `y`, split indices, and the train normalisation stats.
* `metadata.csv`: the dataset table.
* `labels.csv`: id → emotion name.

The final **smoke test** (a quick logistic regression) only verifies that features and labels are aligned. It is *not* the Phase 3 baseline.

---

## 3. Settings used (keep them fixed all project)

| Parameter | Value | Meaning |
|---|---|---|
| `SR` | 22050 Hz | sampling rate |
| `N_FFT` | 2048 | FFT window (samples) |
| `HOP` | 512 | step between frames |
| `N_MFCC` | 40 | coefficients |
| `N_MELS` | 64 | mel bands |
| `DURATION` | 3.0 s | fixed clip length |
| `MAX_FRAMES` | 130 | time steps per clip |

Changing any of these changes every shape downstream, so re-run the whole notebook if you do.

---

## 4. How to run it

1. `pip install librosa numpy scipy pandas matplotlib scikit-learn soundfile joblib`
2. Download RAVDESS (Audio_Speech_Actors_01-24) from Zenodo into `data/RAVDESS/`, and TESS into `data/TESS/`.
3. Open the notebook and *Run All*.

If no dataset is found, the notebook **generates 192 synthetic clips** (6 fake actors × 8 emotions) with RAVDESS-style names so every cell still works. Its 100 % smoke-test accuracy is meaningless: synthetic emotions are trivially separable. Real data will give much lower, honest numbers (typically 50–75 % for simple models on RAVDESS).

Loading the saved data later:

```python
import numpy as np
d = np.load("features/ser_features.npz")
X, y = d["X_mfcc"], d["y"]
train_idx, val_idx, test_idx = d["train_idx"], d["val_idx"], d["test_idx"]
```

---

## 5. Common pitfalls

* **Speaker leakage:** random row-wise splits inflate accuracy. Always split by speaker.
* **Normalising before splitting:** fit statistics on train only.
* **Forgetting dB conversion** on the mel spectrogram: raw power trains badly.
* **Mixing datasets blindly:** RAVDESS (acted, North American, 24 actors) and TESS (2 older female speakers) differ a lot, so check results per dataset.
* **Class imbalance:** look at per-class F1.
* **Variable clip length:** always pad/crop to the same number of frames.

---

## 6. What comes next (Phase 3)

1. Train Logistic Regression, Decision Tree, Random Forest and SVM on `X_flat`; evaluate with accuracy, precision, recall, F1 and a confusion matrix.
2. Reproduce the paper's CNN baseline on MFCC / Mel / Chroma.
3. Compare the `COMBOS` above to select the **optimal feature set** that feeds the CNN-LSTM-Attention model in Phase 4.
