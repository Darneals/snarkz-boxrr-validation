"""
BOXRR-23 user/session selection — v2
=====================================
Change from v1: session/spare picking now guarantees day coverage.

v1 bug (found in audit of its outputs): roles were assigned by rank over an
evenly-spaced index pick, so "spares" were always the two chronologically
last recordings. For users with ~12 eligible recordings, sessions collapsed
to the earliest 10, which for 15/280 users spanned <3 distinct days (1 user's
picked set covered only 2 days). The multi-day-drift criterion was enforced
on the eligible POOL but never asserted on the selected SESSIONS.

v2 pick logic (deterministic, no RNG involved):
  1. Group a user's time-sorted eligible recordings by calendar day.
  2. Choose min(10, n_days) days, evenly spaced across the sorted day list.
  3. Allocate the 10 session slots round-robin across the chosen days
     (busiest days absorb extra slots), picking evenly spaced within a day.
  4. Spares = 2 recordings evenly spaced over the unpicked remainder.
Because user eligibility requires >=3 distinct days, sessions cover >=3 days
by construction. A post-check asserts this and aborts before writing files
if violated.

Eligibility filters, seed, and sampling are UNCHANGED from v1, so the same
280 users are selected; selection_manifest.csv is reproduced identically.
"""

from bson import decode_file_iter
from collections import defaultdict
import numpy as np
import csv, json, sys

PATH        = "replays.bson"
SEED        = 42
N_SELECT    = 280
MIN_RECS    = 12
MIN_DAYS    = 3
MIN_DUR_S   = 60.0
HZ_LO, HZ_HI = 30.0, 200.0
N_SESSIONS  = 10
N_SPARES    = 2

# ── Pass over metadata: eligibility (identical to v1) ─────────────────────────
corrupt = set()
recs = defaultdict(list)   # user_id -> [(timestamp, recording_id, duration)]

with open(PATH, "rb") as f:
    for doc in decode_file_iter(f):
        if doc.get("corrupt_user") == 1:
            corrupt.add(doc["user_id"]); continue
        info = doc.get("info") or {}
        sw = info.get("software") or {}
        if ((sw.get("app") or {}).get("name")) != "Beat Saber":
            continue
        ts = info.get("timestamp")
        dur = doc.get("duration")
        nf = doc.get("num_frames")
        if not isinstance(ts, str) or not dur or not nf:
            continue
        if dur < MIN_DUR_S or not (HZ_LO <= nf / dur <= HZ_HI):
            continue
        recs[doc["user_id"]].append((ts, doc["_id"], dur))

eligible = {}
for uid, lst in recs.items():
    if uid in corrupt or len(lst) < MIN_RECS:
        continue
    days = {ts[:10] for ts, _, _ in lst}
    if len(days) >= MIN_DAYS:
        eligible[uid] = sorted(lst)

print(f"eligible users: {len(eligible):,}")

# ── Sampling (identical to v1 -> same 280 users) ──────────────────────────────
rng = np.random.default_rng(SEED)
uids = sorted(eligible)
chosen = list(rng.choice(uids, size=N_SELECT, replace=False))

# ── v2 session/spare picking: coverage-guaranteed ─────────────────────────────
def pick_sessions(lst, n_sessions=N_SESSIONS, n_spares=N_SPARES):
    days = {}
    for rec in lst:
        days.setdefault(rec[0][:10], []).append(rec)
    day_keys = sorted(days)
    k = min(n_sessions, len(day_keys))
    day_idx = np.unique(np.linspace(0, len(day_keys) - 1, k).round().astype(int))
    chosen_days = [day_keys[i] for i in day_idx]
    alloc = {d: 1 for d in chosen_days}
    remaining = n_sessions - len(chosen_days)
    order = sorted(chosen_days, key=lambda d: -len(days[d]))
    i = 0
    while remaining > 0:
        d = order[i % len(order)]
        if alloc[d] < len(days[d]):
            alloc[d] += 1
            remaining -= 1
        i += 1
        if i > 100000:
            raise RuntimeError("slot allocation stuck (should be impossible: n>=12)")
    sessions = []
    for d in chosen_days:
        rlist = days[d]
        idx = np.unique(np.linspace(0, len(rlist) - 1, alloc[d]).round().astype(int))
        sessions.extend(rlist[j] for j in idx)
    sessions = sorted(sessions)[:n_sessions]
    rest = sorted(set(lst) - set(sessions))
    spares = []
    if rest:
        idx = np.unique(np.linspace(0, len(rest) - 1, min(n_spares, len(rest))).round().astype(int))
        spares = [rest[j] for j in idx]
    return sessions, spares

picks = {}
for uid in chosen:
    picks[uid] = pick_sessions(eligible[uid])

# ── Hard self-verification BEFORE writing anything ────────────────────────────
violations = []
for uid, (sessions, spares) in picks.items():
    sdays = {ts[:10] for ts, _, _ in sessions}
    if len(sessions) != N_SESSIONS:
        violations.append((uid, f"{len(sessions)} sessions"))
    if len(sdays) < MIN_DAYS:
        violations.append((uid, f"sessions span only {len(sdays)} day(s)"))
    if set(sessions) & set(spares):
        violations.append((uid, "session/spare overlap"))
if violations:
    print("SELECTION INVALID — nothing written:", file=sys.stderr)
    for uid, why in violations:
        print(f"  {uid}: {why}", file=sys.stderr)
    sys.exit(1)
print(f"post-check passed: all {len(picks)} users have {N_SESSIONS} sessions "
      f"spanning >= {MIN_DAYS} distinct days")

# ── Outputs ───────────────────────────────────────────────────────────────────
with open("selection_manifest.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["user_id", "n_eligible", "span_days", "first_ts", "last_ts"])
    for uid in chosen:
        lst = eligible[uid]
        days = {ts[:10] for ts, _, _ in lst}
        w.writerow([uid, len(lst), len(days), lst[0][0], lst[-1][0]])

with open("selected_recordings_v2.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["user_id", "recording_id", "timestamp", "duration_s", "role"])
    for uid in chosen:
        sessions, spares = picks[uid]
        for ts, rid, dur in sessions:
            w.writerow([uid, rid, ts, f"{dur:.1f}", "session"])
        for ts, rid, dur in spares:
            w.writerow([uid, rid, ts, f"{dur:.1f}", "spare"])

with open("selection_rule.json", "w") as f:
    json.dump({
        "version": 2,
        "source": "replays.bson (HF mirror cschell/boxrr-23 metadata)",
        "app": "Beat Saber", "exclude_corrupt_users": True,
        "recording_eligibility": {"timestamp_required": True,
            "min_duration_s": MIN_DUR_S, "implied_hz_range": [HZ_LO, HZ_HI]},
        "user_eligibility": {"min_eligible_recordings": MIN_RECS,
            "min_distinct_days": MIN_DAYS},
        "eligible_user_count": len(eligible),
        "sampling": {"method": "uniform_without_replacement",
            "n_selected": N_SELECT, "seed": SEED,
            "note": "uids sorted before rng.choice for reproducibility"},
        "session_pick": ("v2: min(10, n_days) days chosen evenly spaced across the "
            "user's sorted day list; 10 session slots allocated round-robin across "
            "chosen days (evenly spaced within day); 2 spares evenly spaced over "
            "the remainder. Guarantees sessions span >= 3 distinct days; asserted "
            "by post-check before output."),
        "v1_deviation_note": ("v1 assigned spares as the two chronologically last "
            "picks, allowing sessions to collapse onto <3 days for 15/280 users; "
            "v1 outputs discarded, never used for downloads or analysis.")
    }, f, indent=2)

print("wrote: selection_manifest.csv, selected_recordings_v2.csv, selection_rule.json")
