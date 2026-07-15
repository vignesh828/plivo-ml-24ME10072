"""Skeleton: prosodic features + classifier. Runs as-is, scores poorly ON
PURPOSE. Your hour goes into extract_features() and what you learn from
your errors.

    python train.py --data_dir eot_data/english --out predictions.csv
"""
import argparse
import csv
import os
import joblib

import numpy as np
from sklearn.ensemble import RandomForestClassifier # Changed import
from sklearn.model_selection import GroupShuffleSplit

from features import load_wav, speech_before, frame_energy_db, f0_contour
from features import HOP_MS

def _voiced_runs(f0):
    """Contiguous voiced regions -> list of (start_idx, end_idx)."""
    runs, i, n = [], 0, len(f0)
    while i < n:
        if f0[i] > 0:
            j = i
            while j < n and f0[j] > 0:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs

def _slope(y, hop_s):
    if len(y) < 2:
        return 0.0
    t = np.arange(len(y)) * hop_s
    return float(np.polyfit(t, y, 1)[0])

def _trailing_silence_s(e, hop_s, thresh_db=None):
    if len(e) == 0:
        return 0.0
    if thresh_db is None:
        thresh_db = np.median(e) - 15.0
    n = 0
    for v in e[::-1]:
        if v < thresh_db:
            n += 1
        else:
            break
    return n * hop_s

def extract_features(x, sr, pause_start, window_s=1.5):
    hop_s = HOP_MS / 1000.0
    seg = speech_before(x, sr, pause_start, window_s=window_s)
    if len(seg) < sr // 10:
        return np.zeros(20, dtype=np.float32) 

    e = frame_energy_db(seg, sr)
    f0 = f0_contour(seg, sr)
    voiced = f0[f0 > 0]
    runs = _voiced_runs(f0)

    energy_mean, energy_std = float(np.mean(e)), float(np.std(e))
    energy_min, energy_max = float(np.min(e)), float(np.max(e))
    energy_range = energy_max - energy_min
    energy_last5 = float(np.mean(e[-5:])) if len(e) >= 5 else energy_mean
    n_local = max(2, int(0.3 / hop_s))
    energy_slope_local = _slope(e[-n_local:], hop_s)
    trailing_silence_s = _trailing_silence_s(e, hop_s)

    
    if len(voiced):
        pitch_mean, pitch_std = float(np.mean(voiced)), float(np.std(voiced)) or 1e-6
        pitch_min, pitch_last3 = float(np.min(voiced)), float(np.mean(voiced[-3:]))
    else:
        pitch_mean = pitch_std = pitch_min = pitch_last3 = 0.0

    voiced_idx = np.where(f0 > 0)[0]
    if len(voiced_idx) >= 2:
        pitch_slope_global = float(np.polyfit(voiced_idx * hop_s, f0[voiced_idx], 1)[0])
    else:
        pitch_slope_global = 0.0

    
    pitch_final_z = (pitch_last3 - pitch_mean) / pitch_std if pitch_std else 0.0

    
    voiced_ratio = len(voiced) / max(len(f0), 1)
    n_runs = len(runs)
    run_durs = [(b - a) * hop_s for a, b in runs]
    mean_run_dur = float(np.mean(run_durs)) if run_durs else 0.0
    speaking_rate = n_runs / (len(seg) / sr)

    
    turn_elapsed = pause_start
    full_hist = x[: int(pause_start * sr)]
    if len(full_hist) >= sr // 5:
        e_hist = frame_energy_db(full_hist, sr)
        turn_energy_mean = float(np.mean(e_hist))
        turn_energy_std = float(np.std(e_hist)) + 1e-6
    else:
        turn_energy_mean, turn_energy_std = energy_mean, 1.0
    energy_last5_z = (energy_last5 - turn_energy_mean) / turn_energy_std

    return np.array([
        energy_mean, energy_std, energy_min, energy_max, energy_range,
        energy_last5, energy_slope_local, trailing_silence_s,
        pitch_mean, pitch_std, pitch_min, pitch_last3, pitch_slope_global, pitch_final_z,
        voiced_ratio, n_runs, mean_run_dur, speaking_rate,
        energy_last5_z, turn_elapsed,
    ], dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(args.data_dir, "labels.csv"))))
    cache = {}
    X, y, groups, keys = [], [], [], []
    for r in rows:
        path = os.path.join(args.data_dir, r["audio_file"])
        if path not in cache:
            cache[path] = load_wav(path)
        x, sr = cache[path]
        X.append(extract_features(x, sr, float(r["pause_start"])))
        y.append(1 if r["label"] == "eot" else 0)
        groups.append(r["turn_id"])
        keys.append((r["turn_id"], r["pause_index"]))
    X, y = np.array(X), np.array(y)

    
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
                  .split(X, y, groups))
    
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42, class_weight="balanced")
    clf.fit(X[tr], y[tr])
    
    print(f"held-out turn accuracy: {clf.score(X[te], y[te]):.3f} "
          f"(chance ~ {max(np.mean(y), 1-np.mean(y)):.3f})")

    clf.fit(X, y)
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, 'models/eot_model.joblib')
    print("Model saved to models/eot_model.joblib")

    p = clf.predict_proba(X)[:, 1]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), pi_p in zip(keys, p):
            w.writerow([tid, pi, f"{pi_p:.4f}"])
    print(f"wrote {len(keys)} predictions -> {args.out}")
    print("NOTE for your final predict.py: it must load a SAVED model and "
          "predict on unseen data, not refit like this sanity script.")

if __name__ == "__main__":
    main()