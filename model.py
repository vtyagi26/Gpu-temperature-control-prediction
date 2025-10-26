import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --- Model Configuration ---
TIMESTEPS = 30      # 30 seconds of past data
NUM_FEATURES = 5    # 0:temp, 1:power, 2:fan, 3:util, 4:clock
LAMBDA_PHYSICS = 1.0    # Can be 1.0 now that losses are scaled
DELTA_T = 1.0           # seconds between samples
T_AMBIENT = 25.0        # ambient temperature (°C)

# --- De-normalization Constants ---
DENORM_TEMP = 100.0
DENORM_POWER = 300.0
DENORM_FAN = 100.0
DENORM_UTIL = 100.0
DENORM_CLOCK = 2000.0

class PhysicsInformedLSTM(Model):
    def __init__(self, timesteps, num_features, **kwargs):
        super(PhysicsInformedLSTM, self).__init__(**kwargs)
        self.timesteps = timesteps
        self.num_features = num_features

        # --- Learnable physical parameters ---
        self.C_thermal = self.add_weight(
            name='C_thermal_capacitance',
            shape=(1,),
            initializer=tf.keras.initializers.Constant(10.0),
            trainable=True
        )
        self.k_transfer = self.add_weight(
            name='k_heat_transfer',
            shape=(1,),
            initializer=tf.keras.initializers.Constant(0.5),
            trainable=True
        )

        # --- Neural architecture ---
        self.lstm = LSTM(64, activation='tanh', input_shape=(timesteps, num_features), return_sequences=False)
        self.dense1 = Dense(32, activation='tanh') 
        self.out_temp = Dense(1, name='temp_output')

    def call(self, inputs, training=False):
        x = self.lstm(inputs)
        x = self.dense1(x)
        temp_pred_n = self.out_temp(x)
        return temp_pred_n

    def compute_physics_loss(self, X_batch, T_pred_n):
        # 1. Get NORMALIZED inputs
        T_prev_n = X_batch[:, -1, 0:1]
        P_in_n = X_batch[:, -1, 1:2]
        Fan_in_n = X_batch[:, -1, 2:3]  # This is the 0-1 normalized fan speed

        # 2. DE-NORMALIZE to physical units
        T_prev = T_prev_n * DENORM_TEMP
        P_in = P_in_n * DENORM_POWER
        # --- !!! THIS IS THE FIX !!! ---
        # We do NOT de-normalize the fan speed. We use the 0-1 value Fan_in_n
        # directly in the physics equation, just like the simulator.
        T_pred = T_pred_n * DENORM_TEMP

        # 3. Apply physics equation
        dT_dt = (T_pred - T_prev) / DELTA_T
        
        # Use the normalized (0-1) fan speed
        P_out = self.k_transfer * Fan_in_n * (T_pred - T_AMBIENT)
        
        # 'residual' is in physical units (Watts)
        residual = (self.C_thermal * dT_dt) - P_in + P_out

        # 4. NORMALIZE the residual loss
        normalized_residual = residual / DENORM_POWER
        physics_loss = tf.reduce_mean(tf.square(normalized_residual))
        
        return physics_loss

    def train_step(self, data):
        X_batch, y_batch_n = data
        
        with tf.GradientTape() as tape:
            T_pred_n = self(X_batch, training=True)
            data_loss = tf.reduce_mean(tf.square(y_batch_n - T_pred_n))
            physics_loss = self.compute_physics_loss(X_batch, T_pred_n)
            total_loss = data_loss + (LAMBDA_PHYSICS * physics_loss)

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.C_thermal.assign(tf.clip_by_value(self.C_thermal, 0.001, 1000))
        self.k_transfer.assign(tf.clip_by_value(self.k_transfer, 0.001, 1000))

        return {"total_loss": total_loss,
                "data_loss": data_loss,
                "physics_loss": physics_loss,
                "C_thermal": self.C_thermal,
                "k_transfer": self.k_transfer}

    def test_step(self, data):
        X_batch, y_batch_n = data
        T_pred_n = self(X_batch, training=False)
        data_loss = tf.reduce_mean(tf.square(y_batch_n - T_pred_n))
        physics_loss = self.compute_physics_loss(X_batch, T_pred_n)
        total_loss = data_loss + (LAMBDA_PHYSICS * physics_loss)
        
        return {"total_loss": total_loss,
                "data_loss": data_loss,
                "physics_loss": physics_loss}

# --- Main execution (no changes below this line) ---
if __name__ == "__main__":
    
    print("--- Loading Data ---")
    try:
        data = np.load("gpu_sequences.npz")
        X = data["X"]
        y = data["y"].reshape(-1, 1)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure 'gpu_sequences.npz' exists and was created with 5 features.")
        exit()

    if X.shape[2] != NUM_FEATURES:
        print(f"Data shape error! Expected {NUM_FEATURES} features, but got {X.shape[2]}.")
        print("Please regenerate 'gpu_sequences.npz' with the correct features.")
        exit()
        
    print(f"Data loaded successfully: X shape {X.shape}, y shape {y.shape}")

    # Split train / val / test (80/10/10)
    n = len(X)
    split_train = int(0.8 * n)
    split_val = int(0.9 * n)

    X_train, y_train = X[:split_train], y[:split_train]
    X_val, y_val = X[split_train:split_val], y[split_train:split_val]
    X_test, y_test = X[split_val:], y[split_val:]

    print(f"Train samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # Build and compile model
    model = PhysicsInformedLSTM(TIMESTEPS, NUM_FEATURES)
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-4)) # e.g., 5e-4

    # Train
    print("\n--- Starting Model Training ---")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        verbose=1
    )

    print("\n✅ Training complete.")
    
    # --- FIX: Define the "True" constants from the simulator ---
    # These values are from datalog.py, used for comparison
    C_THERMAL_TRUE = 120.0
    K_TRANSFER_TRUE = 0.5
    
    # Now we can compare the learned values to the "true" simulator values!
    print(f"True C_thermal: {C_THERMAL_TRUE} | Learned C_thermal: {float(model.C_thermal.numpy()):.4f}")
    print(f"True k_transfer: {K_TRANSFER_TRUE} | Learned k_transfer: {float(model.k_transfer.numpy()):.4f}")


    # Predict on test set
    preds_n = model.predict(X_test)

    # De-normalize for plotting and metrics
    # --- FIX: Corrected variable name (removed extra underscore) ---
    y_test_actual = y_test * DENORM_TEMP
    preds_actual = preds_n * DENORM_TEMP


    # --- 1. ACCURACY SCORE CHECK ---
    print("\n--- Model Evaluation (Test Set) ---")
    
    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test_actual, preds_actual)
    print(f"Mean Absolute Error (MAE): {mae:.2f} °C")
    print(f"-> On average, the prediction is {mae:.2f}°C off from the actual temperature.")

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_test_actual, preds_actual))
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} °C")
    print(f"-> This penalizes large errors more heavily.")

    # R-squared (R²)
    r2 = r2_score(y_test_actual, preds_actual)
    print(f"R-squared (R²) Score: {r2:.4f}")
    print(f"-> The model explains {r2*100:.1f}% of the variance in the temperature data.")


    # --- 2. ACCURACY PLOTS ---
    print("\nDisplaying prediction plots...")

    # Plot 1: Time Series (Actual vs Predicted over time)
    plt.figure(figsize=(14, 7))
    plt.plot(y_test_actual, label='Actual Temp (°C)', linewidth=1.5, alpha=0.8)
    plt.plot(preds_actual, label='Predicted Temp (°C)', linewidth=1.5, linestyle='--')
    plt.xlabel('Time step (in test set)')
    plt.ylabel('Temperature (°C)')
    plt.title('GPU Temperature Prediction (Time Series)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show() # Show the first plot

    # Plot 2: Scatter Plot (Actual vs Predicted)
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_actual, preds_actual, alpha=0.5, s=10) # 's=10' for smaller dots
    
    # Add the "perfect prediction" line (y=x)
    min_temp = min(np.min(y_test_actual), np.min(preds_actual)) - 2 # Add a little buffer
    max_temp = max(np.max(y_test_actual), np.max(preds_actual)) + 2
    plt.plot([min_temp, max_temp], [min_temp, max_temp], 'r--', linewidth=2, label='Perfect Prediction (y=x)')
    
    plt.xlabel('Actual Temperature (°C)')
    plt.ylabel('Predicted Temperature (°C)')
    plt.title('Accuracy Plot: Actual vs. Predicted Temperature')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axis('equal') # Ensures the x and y axes have the same scale
    plt.show() # Show the second plot


    # --- Save model ---
    model.save('gpu_temp_predictor_model.keras')
    print("✅ Model saved as 'gpu_temp_predictor_model.keras'")