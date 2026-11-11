# Week 9 Lab: Implicit Feedback with Confidence-Weighted ALS

**Course:** DSA 4060 (Recommender Systems)
**Notebook:** `Week9_Implicit_Feedback_ALS.ipynb`

## Overview

This notebook shows how to build a music track recommender from **implicit feedback** (plays, repeat plays, saves) instead of explicit ratings. It uses a synthetic event log and the **Alternating Least Squares (ALS)** algorithm from the `implicit` library.

The lab goes from raw events to ranked Top-5 recommendations, then evaluates the model with a time-aware split.

## Pipeline

1. **Set up:** import libraries and fix the random seed (`4060`).
2. **Create an event log:** 12 users, 18 tracks, 194 events (`play`, `repeat_play`, `save`).
3. **Interaction strength:** map events to weights (play = 1, repeat_play = 3, save = 5) and sum per user-item pair.
4. **Preference and confidence:** preference is 0/1; confidence = `1 + alpha * strength` with `alpha = 20`.
5. **Sparse matrix:** build a 12 x 18 user-item `csr_matrix` of strengths.
6. **Train ALS:** `factors=16`, `regularization=0.10`, `iterations=25`.
7. **Top-5 recommendations:** recommend unseen tracks for user `U01`.
8. **One-variable experiment:** change one setting, retrain, compare with the baseline.
9. **Evaluation:** simulated time split, with Recall@5 compared against a popularity baseline.
10. **Exit ticket:** short written reflection.

## Setup

Requires Python 3.9+ (tested on Python 3.12).

```bash
pip install numpy pandas scipy implicit jupyter
```

Then open the notebook, select your kernel, and run the cells in order:

```bash
jupyter notebook Week9_Implicit_Feedback_ALS.ipynb
```

> **Tip:** If you see an OpenBLAS threadpool warning, set `OPENBLAS_NUM_THREADS=1` before launching Jupyter. It is only a warning and does not break the results.

## Results

| Item | Value |
|---|---|
| Events generated | 194 |
| Matrix shape | 12 users x 18 items |
| Observed interactions | 72 |
| Density | 0.33 |
| ALS Recall@5 | 0.667 |
| Popularity baseline Recall@5 | 0.583 |

- **Top-5 for U01:** Track_05, Track_04, Track_09, Track_07, Track_12.
- **Experiment:** the experiment run gave the same Top-5 as the baseline, in the same order.
- **Caution:** with only 12 users, these numbers are very noisy. Use them to learn the *method*, not to judge the model.

## Key Ideas

- **Implicit vs explicit feedback:** behaviour (plays, saves) versus direct ratings.
- **No interaction is not dislike:** it can mean the user never saw the item.
- **Preference vs confidence:** preference says *whether* a user showed interest; confidence says *how much* to trust it.
- **Scores are not probabilities:** they are only useful for ranking.
- **Time-aware evaluation:** a random split leaks future events into training.

## Limitations

- The data is synthetic and tiny, so results are noisy.
- Event weights are modeling choices, not universal truths.
- Time order is simulated from row order, since there are no real timestamps.
- An offline score does not prove that users benefit.

