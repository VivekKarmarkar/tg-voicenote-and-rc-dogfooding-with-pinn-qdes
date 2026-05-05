import json
import pandas as pd
from pathlib import Path

SEGMENTS_DIR = Path(__file__).parent / "segments"
OUTPUT_DIR = Path(__file__).parent / "test_files"
OUTPUT_DIR.mkdir(exist_ok=True)

with open(SEGMENTS_DIR / "segment_23.json") as f:
    data = json.load(f)

gyr_df = pd.DataFrame({
    "time": data["time"],
    "Gyr_X": data["gyr_x"],
    "Gyr_Y": data["gyr_y"],
    "Gyr_Z": data["gyr_z"],
})
gyr_df.to_csv(OUTPUT_DIR / "gyr_data_seg23.csv", index=False)

meta_df = pd.DataFrame([{
    "q_init_w": 0.0509361714166126,
    "q_init_x": -0.555871643580193,
    "q_init_y": -0.3393518394925133,
    "q_init_z": 0.7571344341250899,
    "q_final_w": -0.20157458208707,
    "q_final_x": -0.7378551037198243,
    "q_final_y": -0.254136757986071,
    "q_final_z": 0.5919054333348864,
}])
meta_df.to_csv(OUTPUT_DIR / "metadata_seg23.csv", index=False)

print(f"Saved: {OUTPUT_DIR / 'gyr_data_seg23.csv'} ({len(gyr_df)} rows)")
print(f"Saved: {OUTPUT_DIR / 'metadata_seg23.csv'} (1 row, 8 columns)")
