"""Audio utilities for the EOT assignment.
Updated to include discriminative prosodic features for EOT detection.
"""
import numpy as np
import soundfile as sf
import librosa

FRAME_MS = 25
HOP_MS = 10

def load_wav(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr

def speech_before(x, sr, pause_start, window_s=1.5):
    """The last `window_s` seconds of audio strictly before the pause."""
    end = int(pause_start * sr)
    start = max(0, end - int(window_s * sr))
    return x[start:end]

def frames(x, sr, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    fl = int(sr * frame_ms / 1000)
    hp = int(sr * hop_ms / 1000)
    if len(x) < fl:
        return np.empty((0, fl), dtype=np.float32)
    n = 1 + (len(x) - fl) // hp
    idx = np.arange(fl)[None, :] + hp * np.arange(n)[:, None]
    return x[idx]

def frame_energy_db(x, sr):
    fr = frames(x, sr)
    rms = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)

def f0_contour(x, sr, frame_ms=40, hop_ms=HOP_MS):
    fr = frames(x, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    return np.array([autocorr_f0(f, sr) for f in fr], dtype=np.float32)

def autocorr_f0(frame, sr, fmin=60.0, fmax=400.0, voicing_thresh=0.30):
    frame = frame - np.mean(frame)
    if np.max(np.abs(frame)) < 1e-4: return 0.0
    ac = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
    ac = ac / ac[0]
    lo = int(sr / fmax)
    hi = min(int(sr / fmin), len(ac) - 1)
    if hi <= lo: return 0.0
    lag = lo + int(np.argmax(ac[lo:hi]))
    return float(sr / lag) if ac[lag] > voicing_thresh else 0.0

def extract_features(x, sr, pause_start, window_s=1.5):
    """
    Extracts discriminative features for EOT detection.
    Optimized to reduce delay by capturing pitch and energy dynamics.
    """
    hop_s = HOP_MS / 1000.0
    seg = speech_before(x, sr, pause_start, window_s=window_s)
    
    if len(seg) < sr // 10:
        return np.zeros(12, dtype=np.float32)

    # 1. Energy Features
    e = frame_energy_db(seg, sr)
    energy_mean = np.mean(e)
    # Energy Slope: negative value here strongly suggests speech ending
    energy_slope = np.polyfit(np.arange(len(e)), e, 1)[0] if len(e) > 1 else 0.0
    # Energy Drop: Compare recent vs total energy
    energy_drop = np.mean(e[-10:]) - np.mean(e) if len(e) > 10 else 0.0

    # 2. Pitch Features
    f0 = f0_contour(seg, sr)
    voiced = f0[f0 > 0]
    pitch_mean = np.mean(voiced) if len(voiced) > 0 else 0.0
    # Pitch Slope: Falling pitch at the end is a classic EOT cue
    pitch_slope = np.polyfit(np.arange(len(voiced)), voiced, 1)[0] if len(voiced) > 1 else 0.0
    # Pitch Acceleration: Is the fall getting faster?
    pitch_accel = (voiced[-1] - voiced[-2]) - (voiced[-2] - voiced[-3]) if len(voiced) > 3 else 0.0

    # 3. Voicing & Rate
    voiced_ratio = len(voiced) / max(len(f0), 1)
    # Zero Crossing Rate (ZCR) - detects breathing vs speech
    zcr = np.mean(librosa.feature.zero_crossing_rate(seg))

    return np.array([
        energy_mean, energy_slope, energy_drop,
        pitch_mean, pitch_slope, pitch_accel,
        voiced_ratio, zcr,
        len(seg) / sr, # duration
        np.std(e), np.max(e), np.std(voiced) if len(voiced) > 0 else 0
    ], dtype=np.float32)