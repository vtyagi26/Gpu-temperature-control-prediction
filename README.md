# GPU Thermal Regulation and Prediction

A machine learning and physics-informed system designed to monitor, predict, and regulate GPU temperature for efficient thermal management, ensuring optimal performance and preventing throttling or hardware damage.


## 📖 Overview

This project implements a **Physics-Informed Neural Network (PINN)** to model and predict GPU thermal behavior using real-time telemetry such as power draw, fan speed, utilization, and clock frequency.  
By combining thermodynamic equations with recurrent neural networks (LSTM), the system learns temperature evolution patterns and dynamically estimates physical parameters such as heat transfer coefficient and thermal capacitance.  
The ultimate goal is to enable **energy-efficient cooling control** in high-performance GPUs used in data centers, gaming, and compute-intensive workloads.


## ✨ Features

- **Real-Time Monitoring:** Continuously tracks GPU telemetry — temperature, fan speed, power draw, and utilization.
- **Physics-Guided Learning:** Integrates thermodynamic constraints into the neural network’s loss function for physically consistent predictions.
- **Future Temperature Prediction:** Predicts GPU temperature 1 second ahead to anticipate and prevent overheating.
- **Custom Fan Control (Planned):** Automatically regulates fan or pump speed to maintain target temperature with minimal energy waste.
- **Data Logging:** Stores telemetry and prediction logs in structured CSV or database format for further analysis.
- **Visualization Tools:** Provides real-time plots of temperature trends, model predictions, and error metrics.


## 🔧 Installation

### Prerequisites
- Python 3.8 or higher  
- NVIDIA GPU drivers (if applicable)  
- CUDA Toolkit (for GPU acceleration)  
- TensorFlow / Keras  
- NumPy, Pandas, Matplotlib  

### Setup

1. **Clone the repository:**
   git clone https://github.com/vtyagi26/Gpu-temperature-control-prediction.git
   cd Gpu-temperature-control-prediction

2. **Create a virtual environment (recommended):**
   python -m venv venv
   source venv/bin/activate       # On Windows: venv\Scripts\activate

3. **Install dependencies:**
   pip install -r requirements.txt


## 🚀 Usage

### 1. **Run the Model Training**
Train the Physics-Informed Neural Network (PINN) on synthetic or collected telemetry data:
python model.py

This will:
- Load the dataset from the configured path.
- Train the model with the physics-based loss function.
- Save the trained model as `gpu_temp_model.h5` for later use.

### 2. **Visualize Predictions**
Use the notebook or script to visualize predicted vs actual GPU temperature:
python plot_results.py

The script will generate real-time comparison graphs of model output and true values.

### 3. **Monitor GPU Temperature (Optional)**
To only log GPU parameters in real-time:
python monitor.py --interval 1

This command captures telemetry data every second using `nvidia-smi` or equivalent APIs.

---

## 📊 Model Details

- **Architecture:** LSTM-based recurrent neural network with dense post-processing layers.
- **Loss Function:** Combined Mean Squared Error (MSE) + Physics regularization term enforcing the RC thermal equation.
- **Target Variable:** GPU core temperature (°C)
- **Input Features:** Power, Fan Speed, Utilization, Clock Speed, Ambient Temp.
- **Performance:** Achieved Mean Absolute Error (MAE) = 0.45°C, R² = 0.986 on synthetic GPU workload simulations.

---

## 🧠 Research Focus

This work lays the foundation for **thermal-aware GPU control systems** integrating physics with machine learning for energy-efficient operation.  
Future extensions include:
- Real-time dynamic control of fan/pump speed.
- Integration with reinforcement learning for adaptive cooling.
- Application to data center-scale GPU clusters.

---

## 📄 License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## 👨‍💻 Author
**Vaibhav Tyagi**  
