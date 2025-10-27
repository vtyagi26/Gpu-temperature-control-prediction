import pandas as pd
import time, datetime, random
import numpy as np

LOG_FILE = 'gpu_telemetry_sim.csv'
LOG_INTERVAL_SECONDS = 1
TOTAL_STEPS = 2500

C_THERMAL_TRUE = 120.0  
K_TRANSFER_TRUE = 0.5   
T_AMBIENT = 25.0       
DELTA_T = LOG_INTERVAL_SECONDS


current_temp = T_AMBIENT + 20
current_load = 10.0

data_log = []

def simulate_gpu_physics(prev_temp, load):
    """
    Simulates the *next* timestep based on the *previous* timestep
    using the equation: C * dT/dt = P_in - P_out
    """
    

    power_in = 20.0 + (load * 1.5) + random.uniform(-5, 5)
    

    fan_speed = max(0, min(100, (prev_temp - 40) * 2.0))
    
    power_out = K_TRANSFER_TRUE * (fan_speed / 100.0) * (prev_temp - T_AMBIENT)
    
    dTemp_dt = (power_in - power_out) / C_THERMAL_TRUE
    
    new_temp = prev_temp + (dTemp_dt * DELTA_T) + random.uniform(-0.1, 0.1)
    
    clock_graphics = 300 + (load * 15) + random.uniform(-10, 10)
    clock_mem = 5000 + (load * 10) + random.uniform(-50, 50)

    return {
        'timestamp': datetime.datetime.now(),
        'temp_c': round(new_temp, 1),
        'power_w': round(power_in, 2),
        'util_gpu_%': int(load),
        'util_mem_%': int(load * random.uniform(0.7, 1.0)),
        'fan_speed_%': int(fan_speed),
        'clock_graphics_mhz': int(clock_graphics),
        'clock_mem_mhz': int(clock_mem)
    }

def get_next_load(prev_load):

    if random.random() < 0.8: # 80% chance to stay
        new_load = prev_load + random.uniform(-10, 10)
    else: # 20% chance to jump
        new_load = random.uniform(5, 100)
    
    # Clamp between 0 and 100
    return max(0, min(100, new_load))


print(f"Starting physical simulation for {TOTAL_STEPS} steps...")
try:
    for i in range(TOTAL_STEPS):
      
        current_load = get_next_load(current_load)
        
        data = simulate_gpu_physics(current_temp, current_load)
        
        current_temp = data['temp_c']
        
        if (i+1) % 100 == 0:
            print(f"Step {i+1}/{TOTAL_STEPS}... Temp={data['temp_c']}°C, Load={data['util_gpu_%']}%")
        data_log.append(data)

except KeyboardInterrupt:
    print("\nSimulation stopped.")

finally:
    if data_log:
        pd.DataFrame(data_log).to_csv(LOG_FILE, index=False)
        print(f"\nData saved to {LOG_FILE}")
    else:
        print("\nNo data generated.")

