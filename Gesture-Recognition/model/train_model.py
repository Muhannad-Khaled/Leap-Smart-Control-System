# import os
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.utils import shuffle
# from sklearn.metrics import classification_report
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# import joblib
# import pickle

# NORMALIZED_DIR = "processed_data/normalized_data"
# MODEL_PATH = "model/lstm_model.h5"
# ENCODER_PATH = "model/label_encoder.pkl"
# HISTORY_PATH = "model/training_history.pkl"
# WINDOW_SIZE = 50

# def load_sequence_data():
#     X, y = [], []
#     for gesture_name in os.listdir(NORMALIZED_DIR):
#         gesture_path = os.path.join(NORMALIZED_DIR, gesture_name)
#         for set_name in os.listdir(gesture_path):
#             set_path = os.path.join(gesture_path, set_name)
#             for file in os.listdir(set_path):
#                 file_path = os.path.join(set_path, file)
#                 df = pd.read_csv(file_path).dropna()
#                 if df.shape[0] < WINDOW_SIZE:
#                     continue
#                 for start in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_SIZE):
#                     window = df.iloc[start:start + WINDOW_SIZE].values
#                     X.append(window)
#                     y.append(gesture_name)
#     return X, y

# def preprocess_data(X, y):
#     X = np.array(X)
#     le = LabelEncoder()
#     y_encoded = le.fit_transform(y)
#     y_onehot = to_categorical(y_encoded)
#     os.makedirs(os.path.dirname(ENCODER_PATH), exist_ok=True)
#     joblib.dump(le, ENCODER_PATH)
#     print(f"✅ Label encoder saved to {ENCODER_PATH}")
#     return X, y_onehot, le

# def build_model(input_shape, num_classes):
#     model = Sequential([
#         Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
#         Dropout(0.3),
#         Bidirectional(LSTM(64)),
#         Dropout(0.3),
#         Dense(64, activation='relu'),
#         Dense(num_classes, activation='softmax')
#     ])
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# if __name__ == "__main__":
#     print("📥 Loading sequence data...")
#     X_raw, y_raw = load_sequence_data()
#     X, y, le = preprocess_data(X_raw, y_raw)
#     X, y = shuffle(X, y, random_state=42)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
import joblib
import pickle

NORMALIZED_DIR = "processed_data/normalized_data"
MODEL_PATH = "model/lstm_model.h5"
ENCODER_PATH = "model/label_encoder.pkl"
HISTORY_PATH = "model/training_history.pkl"
WINDOW_SIZE = 50

def load_sequence_data():
    windows = []
    for gesture_name in os.listdir(NORMALIZED_DIR):
        gesture_path = os.path.join(NORMALIZED_DIR, gesture_name)
        for set_name in os.listdir(gesture_path):
            set_path = os.path.join(gesture_path, set_name)
            for file in os.listdir(set_path):
                file_path = os.path.join(set_path, file)
                df = pd.read_csv(file_path).dropna()
                for start in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_SIZE):
                    window = df.iloc[start:start + WINDOW_SIZE].values
                    if len(window) == WINDOW_SIZE:
                        windows.append((window, gesture_name))
    
    # ✅ Shuffle the windows before splitting into X and y
    windows = shuffle(windows, random_state=42)
    
    X = [w[0] for w in windows]
    y = [w[1] for w in windows]
    return X, y

def preprocess_data(X, y):
    X = np.array(X)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_onehot = to_categorical(y_encoded)
    os.makedirs(os.path.dirname(ENCODER_PATH), exist_ok=True)
    joblib.dump(le, ENCODER_PATH)
    print(f"✅ Label encoder saved to {ENCODER_PATH}")
    return X, y_onehot, le

def build_model(input_shape, num_classes):
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("📥 Loading sequence data...")
    X_raw, y_raw = load_sequence_data()
    X, y, le = preprocess_data(X_raw, y_raw)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🔧 Building LSTM model...")
    model = build_model((X.shape[1], X.shape[2]), y.shape[1])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss', mode='min')
    ]

    print("🚀 Training model...")
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=50, batch_size=8,
                        callbacks=callbacks)

    print(f"📊 Classification Report:")
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    print(f"💾 Saving training history to {HISTORY_PATH}")
    with open(HISTORY_PATH, 'wb') as f:
        pickle.dump(history.history, f)

    print("✅ Training complete and model saved.")
