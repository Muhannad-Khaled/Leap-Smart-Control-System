import os
import pandas as pd
import numpy as np

INPUT_DIR = "data/normalized/circle_cw"
OUTPUT_DIR = "data/features/circle_cw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_features(df):
    features = {}
    for col in df.columns:
        data = df[col].values
        features[f"{col}_mean"] = np.mean(data)
        features[f"{col}_std"] = np.std(data)
        features[f"{col}_min"] = np.min(data)
        features[f"{col}_max"] = np.max(data)
        features[f"{col}_range"] = np.max(data) - np.min(data)
        features[f"{col}_energy"] = np.sum(data ** 2)
    return features

def process_all_sets():
    for set_name in os.listdir(INPUT_DIR):
        set_path = os.path.join(INPUT_DIR, set_name)
        out_set_path = os.path.join(OUTPUT_DIR, set_name)
        os.makedirs(out_set_path, exist_ok=True)

        for file in os.listdir(set_path):
            file_path = os.path.join(set_path, file)
            df = pd.read_csv(file_path)
            features = extract_features(df)

            feature_df = pd.DataFrame([features])
            feature_df.to_csv(os.path.join(out_set_path, file), index=False)

    print("✅ Feature extraction completed and saved.")

if __name__ == "__main__":
    process_all_sets()
