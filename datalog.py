import pandas as pd
import time, datetime, random

LOG_FILE = 'gpu_telemetry_sim.csv'
LOG_INTERVAL_SECONDS = 1
data_log = []

def simulate_gpu_data():
    base_temp = 45 + random.uniform(-2, 2)
    load = random.randint(10, 100)
    temp = base_temp + (load * 0.4) + random.uniform(-1, 1)
    power = 20 + (load * 1.2) + random.uniform(-5, 5)
    fan_speed = min(100, int((temp - 40) * 1.5))
    clock_graphics = 300 + (load * 15) + random.uniform(-10, 10)
    clock_mem = 5000 + (load * 10) + random.uniform(-50, 50)

    return {
        'timestamp': datetime.datetime.now(),
        'temp_c': round(temp, 1),
        'power_w': round(power, 2),
        'util_gpu_%': load,
        'util_mem_%': int(load * random.uniform(0.7, 1.0)),
        'fan_speed_%': fan_speed,
        'clock_graphics_mhz': int(clock_graphics),
        'clock_mem_mhz': int(clock_mem)
    }

try:
    while True:
        data = simulate_gpu_data()
        print(f"Simulated: Temp={data['temp_c']}C, Power={data['power_w']}W, Fan={data['fan_speed_%']}%")
        data_log.append(data)
        time.sleep(LOG_INTERVAL_SECONDS)

except KeyboardInterrupt:
    print("\nSimulation stopped. Saving data...")

finally:
    pd.DataFrame(data_log).to_csv(LOG_FILE, index=False)
    print(f"Data saved to {LOG_FILE}")
