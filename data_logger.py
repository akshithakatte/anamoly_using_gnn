import csv
from datetime import datetime

# Function to log anomalous data to CSV
def log_anomaly(log_file, features, timestamp):
    with open(log_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp] + features)

# Example use (for testing)
if __name__ == "__main__":
    log_file = "anomalies_log.csv"
    
    # Create CSV file with header if it doesn't exist
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Latency (ms)", "Size (bytes)", "Status"])

    # Simulate logging an anomaly
    log_anomaly(log_file, [100, 500, "Anomalous"], datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Anomaly logged.")
