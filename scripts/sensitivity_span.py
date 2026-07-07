"""Session-span sensitivity: is re-identification driven by session time-span?
Splits the 276-user population at the median session span (66 days) and
compares baseline (eta=0) re-id accuracy between halves.
Result (reported in README): low-span 14.57% (20.1x chance), high-span 13.91%
(19.2x chance) -- span does not drive re-identification.
Requires features_real.csv (produced by extract_features.py). Seed 42."""
import csv, numpy as np
from collections import defaultdict
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

F5 = ["head_yaw_cstd","head_pitch_mean","head_roll_std","head_speed_mean","tremor_rms_mm"]
rows = [r for r in csv.DictReader(open("features_real.csv")) if r["status"]=="ok"]
per_user = defaultdict(lambda: {"session":[], "spare":[]})
for r in rows: per_user[r["user_id"]][r["role"]].append(r)
population = {}
for uid in sorted(per_user):
    d = per_user[uid]
    sess = sorted(d["session"], key=lambda r: r["timestamp"])
    spares = sorted(d["spare"], key=lambda r: r["timestamp"])
    while len(sess) < 10 and spares: sess.append(spares.pop(0))
    if len(sess) >= 10: population[uid] = sorted(sess[:10], key=lambda r: r["timestamp"])

def span_days(recs):
    ts = [datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M") for r in recs]
    return (max(ts)-min(ts)).days
spans = {u: span_days(v) for u,v in population.items()}
med = float(np.median(list(spans.values())))
groups = {"low-span":  [u for u in sorted(population) if spans[u] <= med],
          "high-span": [u for u in sorted(population) if spans[u] >  med]}
print(f"median session span: {med:.0f} days")
for name, uids in groups.items():
    X, y = [], []
    for ui, uid in enumerate(uids):
        for r in population[uid]:
            X.append(np.round(np.array([float(r[f]) for f in F5])*1000).astype(np.int64))
            y.append(ui)
    clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
    acc = cross_val_score(clf, np.array(X), np.array(y), cv=5, scoring="accuracy").mean()
    print(f"{name}: n={len(uids)}, baseline re-id {acc*100:.2f}% "
          f"(chance {100/len(uids):.2f}%, ratio {acc*len(uids):.1f}x)")
