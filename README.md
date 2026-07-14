# SNARKZ Real-Data Validation — BOXRR-23 Companion Study

Evaluation pipeline and results for the real-data validation of the noise-injected
biometric commitment mechanism proposed in *A Threat-Modelled Zero-Knowledge Identity
Specification for Spatial Privacy in Immersive Metaverse Systems* . This repository replays that paper's synthetic churn and
re-identification protocol, unchanged, on real VR motion telemetry from the
[BOXRR-23 corpus](https://rdi.berkeley.edu/metaverse/boxrr-23/) (Nair et al., IEEE
TVCG 2024, DOI 10.1109/TVCG.2024.3372087). A full companion manuscript is in
preparation.

## Headline results

| Experiment | Synthetic (SNARKZ paper) | Real (this study) |
|---|---|---|
| Commitment churn, η=0 | 0.0 | 0.0 |
| Commitment churn, η≥0.5 | 1.0 | 1.0 |
| Natural cross-session churn, η=0 | — | 1.0 |
| Re-identification baseline (η=0) | 99.85% (200 subjects, 8 features) | **9.02%** (276 users, 5 features; chance 0.36%) |
| Re-identification at η=4 | 73.60% | 1.99% |
| 4-feature sensitivity (no tremor), η=0 | — | 7.14% |
| Session-span sensitivity | — | null (20.1× vs 19.2× chance) |

Interpretation: the mechanism-level behaviour (commitment churn under noise
injection) replicates exactly on real telemetry, and natural session drift alone
already churns 100% of commitments. The adversarial results do not replicate: the
calibrated synthetic generator overstates the identifiability of session-level
summary features by an order of magnitude. Both studies evaluate a summary-feature
adversary; attacks on full motion time series achieve far higher accuracy on this
corpus (Nair et al., USENIX Security 2023) and are outside the scope of what
noise-injected summary commitments defend against.

## Pipeline (in execution order)

| Stage | Script | Input | Output |
|---|---|---|---|
| 1. Subject selection | `scripts/select_users_v2.py` | BOXRR-23 metadata dump (`replays.bson`, HF mirror `cschell/boxrr-23`) | `selection/selection_manifest.csv`, `selection/selected_recordings_v2.csv`, `selection_rule.json` |
| 2. Data acquisition | `scripts/download_users.py` | selection CSV | `data/<user_id>/<recording_id>.xror` (3,360 recordings, 280 users) |
| 3. Feature extraction | `scripts/extract_features.py` | data tree | `features_real.csv` (3,329 ok / 31 fail) |
| 4. Evaluation | `scripts/eval_real.py` | `features_real.csv` | `results/eval_real_results.json` + `.csv` |
| 5. Sensitivity | `scripts/sensitivity_span.py` | `features_real.csv` | console report |

All stages are seeded (seed 42) and deterministic: stage 4 was executed
independently on two machines (Windows 11 / venv and Ubuntu 24) with numerically
identical output to full float precision.

## Selection rule (v2)

Population: Beat Saber subset of BOXRR-23 (the metadata dump contains no other
app). Corrupt-flagged users excluded. Recording eligibility: timestamp present,
duration ≥ 60 s, implied frame rate 30–200 Hz. User eligibility: ≥ 12 eligible
recordings on ≥ 3 distinct calendar days. From 40,551 eligible users, 280 sampled
uniformly without replacement (seed 42, IDs sorted before sampling — explicitly
not top-N by activity, which would bias re-identification upward). Per user, 10
sessions chosen with guaranteed multi-day coverage plus 2 spares; a v1 pick rule
that could collapse sessions onto fewer days was found in audit, discarded before
any download, and is documented in `selection_rule.json`.

Final analysis population: 276 users × 10 sessions (31 recordings failed
extraction validity checks; 14 spares promoted chronologically; 4 users unable to
reach 10 sessions were dropped — IDs in `results/eval_real_results.json`).

## Feature definitions (5 of the paper's 8 are derivable; no gaze/blink channels exist)

- `head_yaw_cstd` — circular standard deviation of HMD yaw (deg)
- `head_pitch_mean` — mean HMD pitch (deg)
- `head_roll_std` — standard deviation of HMD roll (deg)
- `head_speed_mean` — mean HMD positional speed (m/s), despiked at 5 m/s
- `tremor_rms_mm` — RMS of 8–12 Hz band-passed HMD position magnitude (mm)

Yaw/pitch/roll are computed from forward/right vectors of the HMD quaternion
(XROR convention: Unity left-handed, y-up, meters). Frames with zero-norm
quaternions are dropped; recordings with < 500 valid frames fail.

## Reproducing

```
pip install pymongo fpzip numpy scipy scikit-learn huggingface_hub
# 1. obtain BOXRR-23 metadata + accept the dataset's Data Use Agreement (see below)
python scripts/select_users_v2.py
python scripts/download_users.py --check && python scripts/download_users.py
python scripts/extract_features.py
python scripts/eval_real.py
python scripts/sensitivity_span.py
```

## What is deliberately NOT in this repository

Raw `.xror` recordings and the per-recording feature table (`features_real.csv`)
are excluded. The BOXRR-23 Data Use Agreement prohibits redistribution, and
per-recording biometric feature vectors keyed to user pseudonyms are subject-level
derived data; publishing them would sit inside that prohibition. Both are exactly
reproducible from the seeded pipeline above after obtaining the corpus from its
official source and accepting its DUA. Aggregate results, selection manifests
(recording identifiers are pointers, not data), and all code are included.

## Ethics

This study links sessions to pseudonymous within-dataset user labels only — the
corpus's sanctioned original research use — and makes no attempt to associate any
recording with a real-world identity. Use of BOXRR-23 is subject to its
[Data Use Agreement](https://rdi.berkeley.edu/metaverse/boxrr-23/); users of this
pipeline must accept it before downloading the corpus.

## Citation

BOXRR-23 corpus:
V. Nair et al., "Berkeley Open Extended Reality Recordings 2023 (BOXRR-23),"
IEEE TVCG 30(5), 2024, DOI 10.1109/TVCG.2024.3372087.

## License

Code: MIT. Result files and figures: CC BY-NC-SA 4.0 (inherited from the BOXRR-23
corpus license family).
