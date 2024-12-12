import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from gnn_model import GNNAnomalyDetector
from traffic_simulation import generate_traffic

def generate_training_data(num_samples=100):
    data_list = []
    traffic_generator = generate_traffic()
    for _ in range(num_samples):
        latency, size = next(traffic_generator)

        # Randomly classify some data points as anomalous
        is_anomalous = torch.rand(1).item() > 0.8  # 20% chance of anomaly
        label = 1 if is_anomalous else 0

        # Create node features
        x = torch.tensor([[latency, size]], dtype=torch.float32)

        # Create edge index dynamically
        edge_index = torch.tensor([[0, 0], [0, 0]], dtype=torch.long)

        # Create Data object
        data = Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))
        data_list.append(data)
    return data_list

def train_gnn():
    # Generate training data
    train_data = generate_training_data(num_samples=500)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    # Define the GNN model
    input_dim = 2  # Latency and size as input features
    hidden_dim = 64
    output_dim = 2  # Normal or Anomalous
    model = GNNAnomalyDetector(input_dim, hidden_dim, output_dim)
    model.train()

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = F.nll_loss

    # Training loop
    for epoch in range(20):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "gnn_model.pth")
    print("Model training complete and saved as 'gnn_model.pth'.")

if __name__ == "__main__":
    train_gnn()

