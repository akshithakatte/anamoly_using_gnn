import time
import random

# Function to simulate traffic generation
def generate_traffic(anomaly_chance=0.3):
    while True:
        # Generate normal traffic
        if random.random() > anomaly_chance:
            latency = random.uniform(50, 200)  # Normal latency between 50 and 200 ms
            size = random.uniform(200, 500)    # Normal size between 200 and 500 bytes
        else:
            # Generate anomalous traffic
            latency = random.uniform(500, 1000)  # Higher latency for anomalies
            size = random.uniform(50, 100)      # Smaller size for anomalies
        
        yield (latency, size)
        time.sleep(0.5)  # Adjust sleep time as needed
