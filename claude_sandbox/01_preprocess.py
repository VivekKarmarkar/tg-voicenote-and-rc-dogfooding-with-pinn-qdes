import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

DATA_DIR = Path(__file__).parent / "dataset"
OUT_DIR = Path(__file__).parent / "segments"
OUT_DIR.mkdir(exist_ok=True)

ref = pd.read_csv(DATA_DIR / "reference.txt")
ref.columns = ref.columns.str.strip()
time_ref = ref["time"].values
qw_ref = ref["quaternion_w"].values
qx_ref = ref["quaternion_x"].values
qy_ref = ref["quaternion_y"].values
qz_ref = ref["quaternion_z"].values
gyr_x_ref = ref["XDI_RateOfTurn_X"].values
gyr_y_ref = ref["XDI_RateOfTurn_Y"].values
gyr_z_ref = ref["XDI_RateOfTurn_Z"].values

dt_ref = np.median(np.diff(time_ref))
print(f"Reference: {len(ref)} samples, dt={dt_ref:.6f}s, duration={time_ref[-1]-time_ref[0]:.2f}s")

omega_mag = np.sqrt(gyr_x_ref**2 + gyr_y_ref**2 + gyr_z_ref**2)

STATIONARY_THRESHOLD = 0.05  # rad/s
MIN_MOVING_SAMPLES = 50
PADDING_SAMPLES = 20

is_moving = omega_mag > STATIONARY_THRESHOLD

segments = []
in_segment = False
start_idx = 0

for i in range(len(is_moving)):
    if is_moving[i] and not in_segment:
        start_idx = i
        in_segment = True
    elif not is_moving[i] and in_segment:
        if i - start_idx >= MIN_MOVING_SAMPLES:
            seg_start = max(0, start_idx - PADDING_SAMPLES)
            seg_end = min(len(is_moving), i + PADDING_SAMPLES)
            segments.append((seg_start, seg_end))
        in_segment = False

if in_segment and len(is_moving) - start_idx >= MIN_MOVING_SAMPLES:
    seg_start = max(0, start_idx - PADDING_SAMPLES)
    seg_end = len(is_moving)
    segments.append((seg_start, seg_end))

print(f"\nFound {len(segments)} non-stationary segments:")
for i, (s, e) in enumerate(segments):
    duration = time_ref[e-1] - time_ref[s]
    print(f"  Segment {i}: indices [{s}:{e}], {e-s} samples, {duration:.2f}s")

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(time_ref, omega_mag, 'b-', alpha=0.7, linewidth=0.5)
axes[0].axhline(y=STATIONARY_THRESHOLD, color='r', linestyle='--', label=f'Threshold={STATIONARY_THRESHOLD} rad/s')
for i, (s, e) in enumerate(segments):
    axes[0].axvspan(time_ref[s], time_ref[e-1], alpha=0.2, color='green', label=f'Seg {i}' if i < 5 else None)
axes[0].set_ylabel('|ω| (rad/s)')
axes[0].set_title('Angular velocity magnitude with detected non-stationary segments')
axes[0].legend()

for j, (label, qcomp) in enumerate(zip(['w', 'x', 'y', 'z'], [qw_ref, qx_ref, qy_ref, qz_ref])):
    axes[1].plot(time_ref, qcomp, label=f'q_{label}', linewidth=0.5)
for i, (s, e) in enumerate(segments):
    axes[1].axvspan(time_ref[s], time_ref[e-1], alpha=0.2, color='green')
axes[1].set_ylabel('Quaternion components')
axes[1].set_xlabel('Time (s)')
axes[1].set_title('Reference quaternion with segments')
axes[1].legend()

plt.tight_layout()
plt.savefig(OUT_DIR / "segmentation_overview.png", dpi=150)
plt.close()
print(f"\nSaved segmentation plot to {OUT_DIR / 'segmentation_overview.png'}")

for i, (s, e) in enumerate(segments):
    seg_data = {
        "time": time_ref[s:e].tolist(),
        "gyr_x": gyr_x_ref[s:e].tolist(),
        "gyr_y": gyr_y_ref[s:e].tolist(),
        "gyr_z": gyr_z_ref[s:e].tolist(),
        "q_w": qw_ref[s:e].tolist(),
        "q_x": qx_ref[s:e].tolist(),
        "q_y": qy_ref[s:e].tolist(),
        "q_z": qz_ref[s:e].tolist(),
        "q_start": [qw_ref[s], qx_ref[s], qy_ref[s], qz_ref[s]],
        "q_end": [qw_ref[e-1], qx_ref[e-1], qy_ref[e-1], qz_ref[e-1]],
    }
    seg_path = OUT_DIR / f"segment_{i:02d}.json"
    with open(seg_path, "w") as f:
        json.dump(seg_data, f)
    print(f"Saved segment {i} to {seg_path}")

print("\nPreprocessing complete.")
