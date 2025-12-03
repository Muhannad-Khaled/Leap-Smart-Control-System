import os
import glob
import pandas as pd
import numpy as np
from scipy.signal import medfilt

def clean_dataframe(df, z_thresh=3, kernel_size=3):
    """
    Clean a single DataFrame by:
    - Removing outliers using Z-score
    - Applying median filter to denoise
    """
    df = df.dropna()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Remove outliers
    z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
    df = df[(z_scores < z_thresh).all(axis=1)]

    # Denoise using median filter
    for col in numeric_cols:
        df[col] = medfilt(df[col], kernel_size=kernel_size)

    return df

def clean_all_csv_files(
    input_root="data",
    output_root="processed_data/cleaned_data",
    z_thresh=3,
    kernel_size=3
):
    """
    Cleans all CSV files under input_root directory:
    - Removes outliers using Z-score
    - Applies median filter to denoise
    - Saves cleaned files to output_root, preserving structure
    """
    os.makedirs(output_root, exist_ok=True)

    gestures = os.listdir(input_root)
    for gesture in gestures:
        gesture_path = os.path.join(input_root, gesture)
        if not os.path.isdir(gesture_path):
            continue

        for subset in os.listdir(gesture_path):
            input_folder = os.path.join(gesture_path, subset)
            output_folder = os.path.join(output_root, gesture, subset)
            os.makedirs(output_folder, exist_ok=True)

            for file in glob.glob(os.path.join(input_folder, '*.csv')):
                try:
                    df = pd.read_csv(file)
                    df = clean_dataframe(df, z_thresh, kernel_size)
                    output_path = os.path.join(output_folder, os.path.basename(file))
                    df.to_csv(output_path, index=False)
                    print(f"✅ Cleaned and saved: {output_path}")
                except Exception as e:
                    print(f"❌ Error processing {file}: {e}")

if __name__ == "__main__":
    clean_all_csv_files()


# import os
# import glob
# import pandas as pd
# import numpy as np
# from scipy.signal import medfilt
# from sklearn.preprocessing import RobustScaler
# import logging

# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# def robust_z_score_outliers(df, thresh=3):
#     numeric_cols = df.select_dtypes(include=[np.number]).columns
#     scaler = RobustScaler()
#     scaled_data = scaler.fit_transform(df[numeric_cols])
#     z_scores = np.abs(scaled_data)
#     return df[(z_scores < thresh).all(axis=1)]

# def adaptive_median_filter(data, min_kernel=3, max_kernel=7):
#     kernel_size = max(min_kernel, min(max_kernel, len(data) // 20))
#     if kernel_size % 2 == 0:
#         kernel_size += 1
#     return medfilt(data, kernel_size=kernel_size)

# def clean_dataframe(df, z_thresh=3):
#     df = df.dropna()
#     df = robust_z_score_outliers(df, z_thresh)
#     if df.empty:
#         logging.warning("DataFrame empty after outlier removal.")
#         return df

#     numeric_cols = df.select_dtypes(include=[np.number]).columns
#     for col in numeric_cols:
#         df[col] = adaptive_median_filter(df[col])

#     return df

# def clean_all_csv_files(
#     input_root="data",
#     output_root="processed_data/cleaned_data",
#     z_thresh=3
# ):
#     os.makedirs(output_root, exist_ok=True)

#     for gesture in os.listdir(input_root):
#         gesture_path = os.path.join(input_root, gesture)
#         if not os.path.isdir(gesture_path):
#             continue

#         for subset in os.listdir(gesture_path):
#             input_folder = os.path.join(gesture_path, subset)
#             output_folder = os.path.join(output_root, gesture, subset)
#             os.makedirs(output_folder, exist_ok=True)

#             for file in glob.glob(os.path.join(input_folder, '*.csv')):
#                 try:
#                     df = pd.read_csv(file)
#                     df = clean_dataframe(df, z_thresh)
#                     if not df.empty:
#                         df.to_csv(os.path.join(output_folder, os.path.basename(file)), index=False)
#                         logging.info(f"Cleaned and saved: {os.path.basename(file)}")
#                     else:
#                         logging.warning(f"File skipped (empty after cleaning): {file}")
#                 except Exception as e:
#                     logging.error(f"Error processing {file}: {e}")

# if __name__ == "__main__":
#     clean_all_csv_files()
