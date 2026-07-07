"""
SNARKZ Phase 2 — Stage 1: feature extraction over the BOXRR-23 selection
=========================================================================
Reads selected_recordings_v2.csv, parses every downloaded .xror under data/,
extracts the 5 derivable features per recording, writes features_real.csv.

Feature definitions (fixed here; audited on the sample recording):
  head_yaw_cstd    circular std of HMD yaw (deg); yaw from forward vector
  head_pitch_mean  mean HMD pitch (deg)
  head_roll_std    std of HMD roll (deg)
  head_speed_mean  mean HMD positional speed (m/s), dt-aware, despiked at 5 m/s
  tremor_rms_mm    RMS of 8-12 Hz band-passed HMD position magnitude (mm);
                   NaN if sample rate cannot resolve the band (fs/2 <= 12 Hz)

Validity per recording: >=500 valid frames (nonzero quaternion) after cleaning.
Resumable: reruns skip recording_ids already present in features_real.csv.

Requires: pip install pymongo fpzip numpy scipy
Run:      python extract_features.py
"""

import bson, fpzip, csv, os, sys, time
import numpy as np
from scipy import signal

CSV_IN   = "selected_recordings_v2.csv"
DATA_DIR = "data"
CSV_OUT  = "features_real.csv"
TREMOR_BAND = (8.0, 12.0)
MIN_VALID_FRAMES = 500

FEATURES = ["head_yaw_cstd", "head_pitch_mean", "head_roll_std",
            "head_speed_mean", "tremor_rms_mm"]
META = ["fs_hz", "duration_s", "frac_dropped", "n_frames"]

# ── Parsing & features ────────────────────────────────────────────────────────
def parse_xror(path):
    doc = bson.decode(open(path, "rb").read())
    devices = doc["info"]["hardware"]["devices"]
    a = np.squeeze(fpzip.decompress(doc["frames"])).astype(np.float64)
    if a.ndim != 2:
        raise ValueError(f"unexpected frame array shape {a.shape}")
    n_axes = sum(len(d["axes"]) for d in devices)
    if a.shape[1] != 1 + n_axes:
        raise ValueError(f"{a.shape[1]} cols, expected {1 + n_axes}")
    t = a[:, 0]
    cols, c = {}, 1
    for d in devices:
        n = len(d["axes"])
        cols[d["joint"]] = a[:, c:c+n]
        c += n
    if "HEAD" not in cols or cols["HEAD"].shape[1] < 7:
        raise ValueError("no 7-axis HEAD device")
    return t, cols["HEAD"]

def quat_to_ypr(q):
    x, y, z, w = q[:,0], q[:,1], q[:,2], q[:,3]
    fx = 2*(x*z + w*y)
    fy = 2*(y*z - w*x)
    fz = 1 - 2*(x*x + y*y)
    ry = 2*(x*y + w*z)
    yaw   = np.degrees(np.arctan2(fx, fz))
    pitch = np.degrees(np.arctan2(fy, np.hypot(fx, fz)))
    roll  = np.degrees(np.arcsin(np.clip(ry, -1, 1)))
    return yaw, pitch, roll

def circ_std_deg(deg):
    r = np.radians(deg)
    R = np.hypot(np.mean(np.sin(r)), np.mean(np.cos(r)))
    return float(np.degrees(np.sqrt(-2*np.log(max(R, 1e-12)))))

def extract(path):
    t, head = parse_xror(path)
    pos, q = head[:, :3], head[:, 3:7]
    valid = np.linalg.norm(q, axis=1) > 0.5
    frac_dropped = float(1 - valid.mean())
    t, pos, q = t[valid], pos[valid], q[valid]
    if len(t) < MIN_VALID_FRAMES:
        raise ValueError(f"only {len(t)} valid frames")
    order = np.argsort(t, kind="stable")   # guard: enforce monotonic time
    t, pos, q = t[order], pos[order], q[order]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    yaw, pitch, roll = quat_to_ypr(q)

    dt = np.diff(t)
    good = dt > 1e-6
    vel = np.linalg.norm(np.diff(pos, axis=0), axis=1)[good] / dt[good]
    vel = vel[vel < 5.0]
    if len(vel) < MIN_VALID_FRAMES // 2:
        raise ValueError("too few velocity samples after despiking")

    fs = 1.0 / np.median(dt)
    tremor = np.nan
    if fs / 2 > TREMOR_BAND[1]:
        tu = np.arange(t[0], t[-1], 1.0/fs)
        pm = np.column_stack([np.interp(tu, t, pos[:,i]) for i in range(3)])
        sos = signal.butter(4, list(TREMOR_BAND), btype="bandpass", fs=fs, output="sos")
        bp = signal.sosfiltfilt(sos, pm, axis=0)
        tremor = float(np.sqrt(np.mean(np.sum(bp**2, axis=1)))) * 1000.0

    feats = {
        "head_yaw_cstd":   circ_std_deg(yaw),
        "head_pitch_mean": float(np.mean(pitch)),
        "head_roll_std":   float(np.std(roll)),
        "head_speed_mean": float(np.mean(vel)),
        "tremor_rms_mm":   tremor,
    }
    meta = {"fs_hz": float(fs), "duration_s": float(t[-1]-t[0]),
            "frac_dropped": frac_dropped, "n_frames": int(len(t))}
    return feats, meta

# ── Runner ────────────────────────────────────────────────────────────────────
def main():
    rows = list(csv.DictReader(open(CSV_IN)))
    done = set()
    if os.path.exists(CSV_OUT):
        done = {r["recording_id"] for r in csv.DictReader(open(CSV_OUT))}
        print(f"resuming: {len(done)} recordings already extracted")

    header = ["user_id", "recording_id", "role", "timestamp",
              *FEATURES, *META, "status"]
    new_file = not os.path.exists(CSV_OUT)
    out = open(CSV_OUT, "a", newline="")
    w = csv.DictWriter(out, fieldnames=header)
    if new_file:
        w.writeheader()

    n_ok = n_fail = 0
    t0 = time.time()
    todo = [r for r in rows if r["recording_id"] not in done]
    print(f"to extract: {len(todo)} of {len(rows)} recordings")
    for i, r in enumerate(todo, 1):
        path = os.path.join(DATA_DIR, r["user_id"], r["recording_id"] + ".xror")
        rec = {"user_id": r["user_id"], "recording_id": r["recording_id"],
               "role": r["role"], "timestamp": r["timestamp"]}
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            feats, meta = extract(path)
            rec.update({k: f"{v:.6f}" if np.isfinite(v) else "NaN"
                        for k, v in feats.items()})
            rec.update({k: f"{v:.4f}" if isinstance(v, float) else v
                        for k, v in meta.items()})
            rec["status"] = "ok"
            n_ok += 1
        except Exception as e:
            for k in FEATURES + META:
                rec[k] = ""
            rec["status"] = f"FAIL: {type(e).__name__}: {e}"
            n_fail += 1
        w.writerow(rec)
        if i % 100 == 0:
            out.flush()
            rate = i / (time.time() - t0)
            print(f"  {i}/{len(todo)}  ({rate:.1f} rec/s, "
                  f"ETA {(len(todo)-i)/rate/60:.1f} min)  ok={n_ok} fail={n_fail}")
    out.close()

    print(f"\nextraction complete: ok={n_ok} fail={n_fail}")
    # per-user session validity summary
    from collections import defaultdict
    per_user = defaultdict(lambda: {"session_ok": 0, "spare_ok": 0, "fails": []})
    for r in csv.DictReader(open(CSV_OUT)):
        u = per_user[r["user_id"]]
        if r["status"] == "ok":
            u["session_ok" if r["role"] == "session" else "spare_ok"] += 1
        else:
            u["fails"].append((r["recording_id"], r["role"]))
    short = {u: d for u, d in per_user.items()
             if d["session_ok"] + min(d["spare_ok"], 10 - d["session_ok"]) < 10}
    need_spares = {u: d for u, d in per_user.items()
                   if d["session_ok"] < 10 and u not in short}
    print(f"users with 10 clean sessions outright: "
          f"{sum(1 for d in per_user.values() if d['session_ok'] == 10)}/{len(per_user)}")
    if need_spares:
        print(f"users needing spare substitution: {len(need_spares)}")
    if short:
        print(f"USERS BELOW 10 USABLE RECORDINGS (report back): {len(short)}")
        for u, d in list(short.items())[:20]:
            print(f"  {u}: sessions_ok={d['session_ok']} spares_ok={d['spare_ok']}")

if __name__ == "__main__":
    main()
