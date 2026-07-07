"""
SNARKZ Phase 2 — Stage 2: churn + re-identification on real BOXRR-23 features
===============================================================================
Exact mirror of the v3 synthetic protocol (seed 42, SHA-256 commitment,
SCALING_FACTOR 1000, k-NN k=3, 5-fold CV with per-fold StandardScaler),
run on features_real.csv produced by extract_features.py.

Population rule: 10 sessions per user; extraction-failed sessions replaced by
that user's ok spares in chronological order; users unable to reach 10 are
dropped (N=276 of 280; dropped IDs recorded in the output JSON).

Additions over the synthetic protocol (both pre-justified):
  - natural cross-session churn at eta=0 (real drift only, no injected noise)
  - 4-feature sensitivity run excluding tremor_rms_mm

Requires: pip install numpy scikit-learn
Run:      python eval_real.py     (deterministic; must reproduce the container
                                   run's numbers exactly given the same
                                   features_real.csv and sklearn >= 1.3)
"""

import csv, json, hashlib
import numpy as np
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

RANDOM_SEED    = 42
SCALING_FACTOR = 1000
ETA_LEVELS     = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
N_CHURN_TRIALS = 200
N_SESSIONS     = 10
F5 = ["head_yaw_cstd", "head_pitch_mean", "head_roll_std",
      "head_speed_mean", "tremor_rms_mm"]
rng = np.random.default_rng(RANDOM_SEED)

rows = [r for r in csv.DictReader(open("features_real.csv")) if r["status"] == "ok"]
per_user = defaultdict(lambda: {"session": [], "spare": []})
for r in rows:
    per_user[r["user_id"]][r["role"]].append(r)

population, promoted, dropped = {}, 0, []
for uid in sorted(per_user):
    d = per_user[uid]
    sess = sorted(d["session"], key=lambda r: r["timestamp"])
    spares = sorted(d["spare"], key=lambda r: r["timestamp"])
    while len(sess) < N_SESSIONS and spares:
        sess.append(spares.pop(0)); promoted += 1
    if len(sess) >= N_SESSIONS:
        population[uid] = sorted(sess[:N_SESSIONS], key=lambda r: r["timestamp"])
    else:
        dropped.append(uid)
N_USERS = len(population)
uids = sorted(population)
print(f"population: {N_USERS} users x {N_SESSIONS} sessions "
      f"({promoted} spares promoted, {len(dropped)} dropped)")

def vec(r, feats): return np.array([float(r[f]) for f in feats])
def quantise(v):   return np.round(v * SCALING_FACTOR).astype(np.int64)
def commitment(sk, bq):
    return hashlib.sha256((str(sk) + "," +
        ",".join(str(int(x)) for x in bq)).encode()).hexdigest()
def inject(bq, eta, rng):
    if eta == 0.0:
        return bq.copy()
    return bq + rng.integers(0, int(round(eta * SCALING_FACTOR)) + 1, size=bq.shape)

# Experiment 1: mechanism-level commitment churn (fixed real vector)
churn = {}
for eta in ETA_LEVELS:
    hits = []
    for _ in range(N_CHURN_TRIALS):
        uid = uids[rng.integers(0, N_USERS)]
        r = population[uid][rng.integers(0, N_SESSIONS)]
        q = quantise(vec(r, F5))
        sk = rng.integers(1_000_000, 9_999_999_999)
        hits.append(1 if commitment(sk, inject(q, eta, rng)) !=
                         commitment(sk, inject(q, eta, rng)) else 0)
    churn[eta] = float(np.mean(hits))
    print(f"  churn eta={eta:.1f}: {churn[eta]:.4f}")

# natural cross-session churn at eta=0
nat = []
for _ in range(N_CHURN_TRIALS):
    uid = uids[rng.integers(0, N_USERS)]
    i, j = rng.choice(N_SESSIONS, size=2, replace=False)
    sk = rng.integers(1_000_000, 9_999_999_999)
    nat.append(1 if commitment(sk, quantise(vec(population[uid][i], F5))) !=
                    commitment(sk, quantise(vec(population[uid][j], F5))) else 0)
nat_churn = float(np.mean(nat))
print(f"natural cross-session churn @ eta=0: {nat_churn:.4f}")

# Experiment 2: re-identification (5-feature primary, 4-feature sensitivity)
def reid(feats, label):
    print(f"re-id {label}:")
    out = {}
    for eta in ETA_LEVELS:
        X, y = [], []
        for ui, uid in enumerate(uids):
            for r in population[uid]:
                X.append(inject(quantise(vec(r, feats)), eta, rng))
                y.append(ui)
        clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
        out[eta] = float(cross_val_score(clf, np.array(X), np.array(y),
                                         cv=5, scoring="accuracy").mean())
        print(f"  eta={eta:.1f} acc={out[eta]:.4f}")
    return out

reid5 = reid(F5, "5-feature")
reid4 = reid(F5[:4], "4-feature (no tremor)")

json.dump({
    "config": {"n_users": N_USERS, "n_sessions": N_SESSIONS, "seed": RANDOM_SEED,
        "scaling_factor": SCALING_FACTOR, "features": F5,
        "n_churn_trials": N_CHURN_TRIALS,
        "population_rule": "10 sessions/user; ok spares promoted chronologically; "
                           f"users below 10 dropped ({len(dropped)}): {dropped}",
        "cv": "StandardScaler in Pipeline (per-fold fit), kNN k=3, 5-fold CV",
        "chance_accuracy": 1.0 / N_USERS},
    "results": {"churn": churn,
        "natural_cross_session_churn_eta0": nat_churn,
        "reid_5feature": reid5, "reid_4feature_no_tremor": reid4}
}, open("eval_real_results.json", "w"), indent=2)
print("saved eval_real_results.json")
