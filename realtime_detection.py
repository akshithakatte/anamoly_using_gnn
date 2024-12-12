import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from gnn_model import GNNAnomalyDetector
from traffic_simulation import generate_traffic
import csv
import time
from datetime import datetime

# Set the device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Load the trained model
input_dim = 2  # Latency and size as input features
hidden_dim = 64
output_dim = 2  # Normal or Anomalous
model = GNNAnomalyDetector(input_dim, hidden_dim, output_dim).to(device)
model.load_state_dict(torch.load("gnn_model.pth"))
model.eval()

# Output CSV file to log anomalies
output_file = "anomalies.csv"

# Initialize the CSV file with headers
with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Latency", "Size", "Anomaly Score"])

# Function for real-time detection
def detect_anomalies():
    traffic_generator = generate_traffic()
    print("Real-time traffic anomaly detection started...")
    
    for _ in range(100):  # Simulate 100 traffic data points
        # Debug: Check traffic generation
        latency, size = next(traffic_generator)
        print(f"Generated traffic: Latency={latency}, Size={size}")  # Add this line

        # Create node features
        x = torch.tensor([[latency, size]], dtype=torch.float32).to(device)
        
        # Create edge index for a simple graph
        edge_index = torch.tensor([[0, 0], [0, 0]], dtype=torch.long).to(device)
        
        # Create Data object
        data = Data(x=x, edge_index=edge_index)
        data = data.to(device)
        
        # Predict using the model
        with torch.no_grad():
            output = model(data)
            anomaly_score = torch.exp(output[0][1]).item()  # Probability of being anomalous
            is_anomalous = anomaly_score > 0.5  # Threshold for anomaly detection
            
            # Debug: Log all predictions
            print(f"Traffic processed: Latency={latency}, Size={size}, "
                  f"Anomaly Score={anomaly_score:.4f}, Anomalous={is_anomalous}")  # Add this line
            
            # Log anomalous instances
            if is_anomalous:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"Anomaly detected at {timestamp}: Latency={latency}, Size={size}, Score={anomaly_score:.4f}")
                
                with open(output_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([timestamp, latency, size, anomaly_score])
        
        # Wait for a short interval to simulate real-time traffic
        time.sleep(0.5)

# Run the detection
if __name__ == "__main__":
    detect_anomalies()
7