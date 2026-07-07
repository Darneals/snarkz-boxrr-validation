"""
BOXRR-23 selected-user downloader
==================================
Reads selected_recordings_v2.csv, downloads each of the 280 user tarballs
from the HF mirror (cschell/boxrr-23), extracts ONLY the selected recordings,
deletes the tarball, and verifies the final tree.

Usage:
    python download_users.py --check          # verify all tarballs exist, no download
    python download_users.py --sample         # process FIRST user only (for the schema sample)
    python download_users.py                  # full run (resumable; rerun after interruption)

Requires:  pip install huggingface_hub
Output:    data/<user_id>/<recording_id>.xror  (+ download_log.csv)
"""

import csv, os, sys, tarfile, time
from collections import defaultdict
from huggingface_hub import HfApi, hf_hub_download

REPO       = "cschell/boxrr-23"
CSV_PATH   = "selected_recordings_v2.csv"
DATA_DIR   = "data"
LOG_PATH   = "download_log.csv"
MAX_RETRY  = 3

CHECK_ONLY = "--check" in sys.argv
SAMPLE     = "--sample" in sys.argv

# ── Load selection ────────────────────────────────────────────────────────────
wanted = defaultdict(set)   # user_id -> {recording_id}
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        wanted[row["user_id"]].add(row["recording_id"])
users = sorted(wanted)
print(f"selection: {len(users)} users, {sum(len(v) for v in wanted.values())} recordings")

# ── Resolve tarball paths (verify BEFORE downloading) ─────────────────────────
api = HfApi()
prefixes = sorted({u[0] for u in users})
repo_files = {}   # user_id -> path_in_repo
print(f"listing repo folders for {len(prefixes)} prefixes...")
for p in prefixes:
    try:
        for entry in api.list_repo_tree(REPO, repo_type="dataset",
                                        path_in_repo=f"users/{p}", recursive=False):
            fname = os.path.basename(entry.path)
            uid = fname.split(".")[0]
            if uid in wanted:
                repo_files[uid] = entry.path
    except Exception as e:
        print(f"  !! listing users/{p} failed: {e}")

missing = [u for u in users if u not in repo_files]
print(f"resolved: {len(repo_files)}/{len(users)} tarballs")
if missing:
    print("MISSING FROM REPO (report these back before proceeding):")
    for u in missing:
        print(f"  {u}")
    sys.exit(1)
if CHECK_ONLY:
    print("--check passed: all tarballs present in repo. No downloads performed.")
    sys.exit(0)

# ── Download → extract selected → delete tar ─────────────────────────────────
os.makedirs(DATA_DIR, exist_ok=True)
log_exists = os.path.exists(LOG_PATH)
log = open(LOG_PATH, "a", newline="")
logw = csv.writer(log)
if not log_exists:
    logw.writerow(["user_id", "n_extracted", "n_expected", "tar_mb", "status"])

def user_done(uid):
    d = os.path.join(DATA_DIR, uid)
    if not os.path.isdir(d):
        return False
    have = {fn.split(".")[0] for fn in os.listdir(d)}
    return wanted[uid] <= have

todo = [u for u in users if not user_done(u)]
if SAMPLE:
    todo = todo[:1]
    print("SAMPLE MODE: processing one user, then stopping.")
print(f"to process: {len(todo)} users ({len(users) - len(todo)} already complete)\n")

for i, uid in enumerate(todo, 1):
    outdir = os.path.join(DATA_DIR, uid)
    os.makedirs(outdir, exist_ok=True)
    status, n_ex, tar_mb = "", 0, 0.0
    for attempt in range(1, MAX_RETRY + 1):
        try:
            tar_path = hf_hub_download(REPO, repo_files[uid], repo_type="dataset")
            tar_mb = os.path.getsize(tar_path) / 1e6
            with tarfile.open(tar_path) as tf:
                for member in tf.getmembers():
                    rid = os.path.basename(member.name).split(".")[0]
                    if rid in wanted[uid]:
                        member.name = os.path.basename(member.name)  # flatten
                        tf.extract(member, outdir)
                        n_ex += 1
            os.remove(tar_path)  # keep peak disk low
            status = "ok" if n_ex == len(wanted[uid]) else f"INCOMPLETE ({n_ex}/{len(wanted[uid])})"
            break
        except Exception as e:
            status = f"error attempt {attempt}: {e}"
            time.sleep(5 * attempt)
    logw.writerow([uid, n_ex, len(wanted[uid]), f"{tar_mb:.1f}", status])
    log.flush()
    print(f"[{i}/{len(todo)}] {uid}  tar={tar_mb:.0f} MB  extracted={n_ex}/{len(wanted[uid])}  {status}")

log.close()

# ── Final verification ────────────────────────────────────────────────────────
total, present = 0, 0
problems = []
for uid, rids in wanted.items():
    d = os.path.join(DATA_DIR, uid)
    have = {fn.split(".")[0] for fn in os.listdir(d)} if os.path.isdir(d) else set()
    total += len(rids)
    present += len(rids & have)
    if not rids <= have:
        problems.append((uid, sorted(rids - have)))

print(f"\nfinal: {present}/{total} selected recordings on disk")
if problems and not SAMPLE:
    print("users with missing recordings (candidates for spare substitution):")
    for uid, miss in problems[:20]:
        print(f"  {uid}: missing {len(miss)}")
    print("Report this list back — spares exist for exactly this case.")
elif not SAMPLE:
    print("all recordings present. Selection tree complete.")
