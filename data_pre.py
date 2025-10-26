import pandas as pd
import numpy as np
import sys
import os

# --- Configuration ---
RAW_CSV_FILE = 'gpu_telemetry_sim.csv' # Make sure this matches your raw log file
OUTPUT_NPZ_FILE = 'gpu_sequences.npz'

# 1. --- FEATURE MISMATCH FIX ---
# We now use all 5 features, in the exact order the model expects.
FEATURES = [
    'temp_c',
    'power_w',
    'fan_speed_%',
    'util_gpu_%',
    'clock_graphics_mhz'
]
TARGET_COLUMN = 'temp_c'

# 2. --- NORMALIZATION MISMATCH FIX ---
# We use simple division, matching the DENORM_* values in model.py
# (No more MinMaxScaler or .pkl file)
NORM_VALUES = {
    'temp_c': 100.0,
    'power_w': 300.0,
    'fan_speed_%': 100.0,
    'util_gpu_%': 100.0,
    'clock_graphics_mhz': 2000.0
}

# 3. --- TIME MISMATCH FIX ---
# We set LOOKBACK to 30 (past) and predict the *very next* timestep
# to match the model's DELTA_T = 1.0.
LOOKBACK = 30
# ---------------------

def preprocess():
    print(f"--- Loading raw data from '{RAW_CSV_FILE}' ---")
    try:
        df = pd.read_csv(RAW_CSV_FILE)
    except FileNotFoundError:
        print(f"❌ Error: '{RAW_CSV_FILE}' not found.")
        print("Please ensure your data log file is named correctly.")
        sys.exit(1)

    # --- 2. Sort by timestamp if available ---
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)

    # --- 3. Select relevant features ---
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"❌ Error: Missing required columns in CSV: {missing}")
        print("Please ensure your logging script captures all 5 features.")
        sys.exit(1)
            
    df_features = df[FEATURES].copy()

    # --- 4. Handle missing or NaN values ---
    # Use ffill/bfill which are the new methods
    df_features = df_features.interpolate(method='linear').ffill().bfill()
    if df_features.isnull().values.any():
        print("❌ Error: Data still contains NaNs after filling. Check raw data.")
        sys.exit(1)

    # --- 5. Normalize features ---
    print("--- Normalizing data using simple division ---")
    for col, norm_val in NORM_VALUES.items():
        if col not in df_features:
            print(f"Warning: '{col}' from NORM_VALUES not in dataframe, skipping.")
            continue
        print(f"Normalizing '{col}' by dividing by {norm_val}")
        df_features[col] = df_features[col] / norm_val
        
    # Clip values to [0, 1] (or slightly higher for robustness)
    df_features = df_features.clip(0.0, 1.0)

    # --- 6. Create Sequences (X) and Targets (y) ---
    print(f"--- Creating sequences with length {LOOKBACK} (predicting 1 step ahead) ---")
    
    feature_data = df_features.to_numpy()
    target_data = df_features[TARGET_COLUMN].to_numpy()

    sequences_X, targets_y = [], []

    # We loop from LOOKBACK until the end of the dataframe
    for i in range(LOOKBACK, len(feature_data)):
        # Input sequence (X) is from [i - LOOKBACK] up to (but not including) [i]
        X_window = feature_data[i - LOOKBACK : i]
        
        # Target (y) is the temperature at step [i] (the *next* step)
        y_target = target_data[i]
        
        sequences_X.append(X_window)
        targets_y.append(y_target)

    if not sequences_X:
        print(f"❌ Error: No sequences created. Your data file has {len(df)} rows,")
        print(f"which is not enough for a LOOKBACK length of {LOOKBACK}.")
        sys.exit(1)

    X = np.array(sequences_X)
    y = np.array(targets_y).reshape(-1, 1)

    # --- 7. Save Preprocessed Data ---
    np.savez(OUTPUT_NPZ_FILE, X=X, y=y)
    print(f"\n✅ Saved preprocessed sequences to {OUTPUT_NPZ_FILE}")
    print(f"📊 X shape: {X.shape}  |  y shape: {y.shape}")
    
    if X.shape[2] != len(FEATURES):
        print(f"--- !!! CRITICAL ERROR !!! ---")
        print(f"Output shape {X.shape} does not have {len(FEATURES)} features.")
        
    # --- 8. Preview Sample ---
    print("\n🔹 Sample input sequence (first 5 timesteps):")
    print(pd.DataFrame(X[0], columns=FEATURES).head())
    print(f"\n🔹 Corresponding target temperature (1s ahead): {y[0][0]:.4f}")

if __name__ == "__main__":
    # Just in case, make sure the old incompatible files are gone
    if os.path.exists('gpu_scaler.pkl'):
        print("Removing old 'gpu_scaler.pkl'...")
        os.remove('gpu_scaler.pkl')
    
    preprocess()