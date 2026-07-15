import os
import csv
import argparse
import joblib
import numpy as np
from features import load_wav, speech_before, frame_energy_db, f0_contour, HOP_MS


def _voiced_runs(f0):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'eot_model.joblib')
    if not os.path.exists(model_path):
        model_path = 'models/eot_model.joblib'
        
    model = joblib.load(model_path)

    # Read the labels.csv
    labels_path = os.path.join(args.data_dir, "labels.csv")
    rows = list(csv.DictReader(open(labels_path)))

    cache = {}
    predictions = []

    for r in rows:
        path = os.path.join(args.data_dir, r["audio_file"])
        if path not in cache:
            cache[path] = load_wav(path)
        x, sr = cache[path]
        
        
        feat = extract_features(x, sr, float(r["pause_start"]))
        
        
        prob = model.predict_proba(feat.reshape(1, -1))[0][1]
        
        predictions.append([r["turn_id"], r["pause_index"], f"{prob:.4f}"])

    
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["turn_id", "pause_index", "p_eot"])
        writer.writerows(predictions)
        
    print(f"Successfully generated {len(predictions)} predictions -> {args.out}")

if __name__ == "__main__":
    main()