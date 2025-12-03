import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# ================= CONFIG =================
INPUT_DIR = "processed_data/cleaned_data"                      # <-- ✅ المسار الجديد للبيانات المنظّفة
OUTPUT_DIR = "processed_data/normalized_data"                  # <-- ✅ مكان حفظ البيانات بعد التطبيع
SCALER_PATH = "model/scaler_global.pkl"
WINDOW_SIZE = 50
# ==========================================

def extract_all_windows():
    all_windows = []
    for gesture in os.listdir(INPUT_DIR):
        gesture_path = os.path.join(INPUT_DIR, gesture)
        if not os.path.isdir(gesture_path):
            continue

        for subset in os.listdir(gesture_path):
            subset_path = os.path.join(gesture_path, subset)
            for file in os.listdir(subset_path):
                if file.endswith(".csv"):
                    path = os.path.join(subset_path, file)
                    try:
                        df = pd.read_csv(path).dropna()
                        for start in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_SIZE):
                            window = df.iloc[start:start + WINDOW_SIZE]
                            if len(window) == WINDOW_SIZE:
                                all_windows.append(window)
                    except Exception as e:
                        print(f"❌ Error processing {path}: {e}")
    return all_windows

def generate_global_scaler():
    print("📥 Collecting all windows...")
    all_windows = extract_all_windows()

    if not all_windows:
        print("❌ No valid windows found.")
        return

    combined_df = pd.concat(all_windows)
    scaler = StandardScaler()
    scaler.fit(combined_df)
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Global scaler saved to {SCALER_PATH}")

def normalize_dataframe(df, scaler_path=SCALER_PATH):
    """
    Normalize a DataFrame using the global pre-fitted scaler.
    """
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"❌ Global scaler not found at: {scaler_path}")
    scaler = joblib.load(scaler_path)
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    normalized = scaler.transform(df[numeric_cols])
    return pd.DataFrame(normalized, columns=numeric_cols)

def normalize_all_with_global_scaler():
    """
    Normalize all files under INPUT_DIR using global scaler
    and save to OUTPUT_DIR, preserving folder structure.
    """
    if not os.path.exists(SCALER_PATH):
        print("⚠️ Global scaler not found. Run generate_global_scaler() first.")
        return

    scaler = joblib.load(SCALER_PATH)

    for gesture in os.listdir(INPUT_DIR):
        gesture_path = os.path.join(INPUT_DIR, gesture)
        if not os.path.isdir(gesture_path):
            continue

        for subset in os.listdir(gesture_path):
            input_folder = os.path.join(gesture_path, subset)
            output_folder = os.path.join(OUTPUT_DIR, gesture, subset)
            os.makedirs(output_folder, exist_ok=True)

            for file in os.listdir(input_folder):
                if file.endswith(".csv"):
                    path = os.path.join(input_folder, file)
                    try:
                        df = pd.read_csv(path).dropna()
                        numeric_cols = df.select_dtypes(include=[float, int]).columns
                        norm_df = pd.DataFrame(scaler.transform(df[numeric_cols]), columns=numeric_cols)
                        norm_df.to_csv(os.path.join(output_folder, file), index=False)
                        print(f"✅ Normalized: {gesture}/{subset}/{file}")
                    except Exception as e:
                        print(f"❌ Error normalizing {path}: {e}")

if __name__ == "__main__":
    generate_global_scaler()
    normalize_all_with_global_scaler()


