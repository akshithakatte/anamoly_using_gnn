import torch
import matplotlib.pyplot as plt
from datetime import datetime
from torch_geometric.data import Data
from gnn_model import GNNAnomalyDetector
from traffic_simulation import generate_traffic
from data_logger import log_anomaly

# Define the device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the GNN model
input_dim = 2  # Latency and size as input features
hidden_dim = 64
output_dim = 2  # Normal or Anomalous

model = GNNAnomalyDetector(input_dim, hidden_dim, output_dim).to(device)
model.load_state_dict(torch.load("gnn_model.pth"))  # Load pre-trained model weights
model.eval()

# Set up CSV log file
log_file = "anomalies_log.csv"

# Real-time data plotting setup
plt.ion()  # Turn on interactive mode for live updates
fig, ax = plt.subplots()
ax.set_title("Real-Time Traffic Anomaly Detection")
ax.set_xlabel("Traffic Instance")
ax.set_ylabel("Feature Value")

# Main loop for real-time traffic monitoring
print("Starting real-time traffic analysis...")
for latency, size in generate_traffic():
    # Create node features
    x = torch.tensor([[latency, size]], dtype=torch.float32).to(device)

    # Create edge index for a self-loop graph
    edge_index = torch.tensor([[0, 0], [0, 0]], dtype=torch.long).to(device)

    # Create Data object
    data = Data(x=x, edge_index=edge_index)
    data = data.to(device)

    # Run the GNN model to predict if the traffic data is normal or anomalous
    with torch.no_grad():
        output = model(data)
        pred = output.argmax(dim=1)  # Predict label (0 for normal, 1 for anomalous)

    # Log anomalies to CSV if detected
    if pred == 1:  # Anomalous traffic detected
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        features = [latency, size, "Anomalous"]
        log_anomaly(log_file, features, timestamp)
        print(f"Anomaly detected at {timestamp} with features {features}")

    # Plot the data for visualization
    ax.clear()
    ax.plot([latency], label="Latency (ms)", color="blue")
    ax.plot([size], label="Size (bytes)", color="green")
    ax.legend()
    plt.pause(0.1)  # Update the plot every 0.1 seconds

plt.ioff()  # Turn off interactive mode after the loop ends
print("Real-time traffic analysis complete.")
