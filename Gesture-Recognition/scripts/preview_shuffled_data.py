import os
import pandas as pd
from sklearn.utils import shuffle

NORMALIZED_DIR = "processed_data/normalized_data"
SHUFFLED_DIR = "scripts/data/shuffled_data"
WINDOW_SIZE = 50

os.makedirs(SHUFFLED_DIR, exist_ok=True)

all_windows = []

for gesture in os.listdir(NORMALIZED_DIR):
    gesture_path = os.path.join(NORMALIZED_DIR, gesture)
    if not os.path.isdir(gesture_path):
        continue

    for subset in os.listdir(gesture_path):
        subset_path = os.path.join(gesture_path, subset)
        for file in os.listdir(subset_path):
            file_path = os.path.join(subset_path, file)
            df = pd.read_csv(file_path)
            for start in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_SIZE):
                window = df.iloc[start:start + WINDOW_SIZE].copy()
                if len(window) == WINDOW_SIZE:
                    window["label"] = gesture
                    all_windows.append(window)

shuffled_windows = shuffle(all_windows, random_state=42)

# Save
for i, window_df in enumerate(shuffled_windows):
    out_path = os.path.join(SHUFFLED_DIR, f"shuffled_sample_{i:04}.csv")
    window_df.to_csv(out_path, index=False)

print(f"✅ Saved {len(shuffled_windows)} shuffled samples to {SHUFFLED_DIR}")
