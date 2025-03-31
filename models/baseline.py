import torch
from torch_geometric.nn import SAGEConv, to_hetero

class GNNEncoder(torch.nn.Module):
    """
    The GNNEncoder class defines a Graph Neural Network encoder using two GraphSAGE convolution layers.
    It processes node features and edge indices to produce node embeddings.
    """
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # First GraphSAGE convolution layer
        self.conv1 = SAGEConv((-1, -1, -1, -1), hidden_channels)
        # Second GraphSAGE convolution layer
        self.conv2 = SAGEConv((-1, -1, -1, -1), out_channels)

    def forward(self, x, edge_index):
        # Apply first convolution and ReLU activation
        x = self.conv1(x, edge_index).relu()
        # Apply second convolution
        x = self.conv2(x, edge_index)
        return x

class EdgeDecoder(torch.nn.Module):
    """
    The EdgeDecoder class defines a decoder for edge prediction.
    It takes node embeddings and edge indices as input and outputs a score for each edge.
    """
    def __init__(self, hidden_channels):
        super().__init__()
        self.user_lin = torch.nn.Linear(hidden_channels, hidden_channels)
        self.restaurant_lin = torch.nn.Linear(hidden_channels, hidden_channels)

    def forward(self, z_dict, edge_label_index):
        # Separate user and restaurant indices
        row, col = edge_label_index
        user_emb = z_dict['user'][row]
        restaurant_emb = z_dict['restaurant'][col]
        return (self.user_lin(user_emb) * self.restaurant_lin(restaurant_emb)).sum(dim=-1)

class BaselineModel(torch.nn.Module):
    """
    Baseline recommendation model using GraphSAGE.
    """
    def __init__(self, metadata, hidden_channels=32, device=None):
        """
        Initialize the model.
        
        Parameters:
        - metadata: Graph metadata for heterogeneous graph transformation
        - hidden_channels: Number of hidden dimensions
        - device: Computation device (CPU/GPU)
        """
        super().__init__()
        # Device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize the GNN encoder
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        # Convert the encoder to handle heterogeneous graphs
        self.encoder = to_hetero(self.encoder, metadata, aggr='sum')
        # Initialize the edge decoder
        self.decoder = EdgeDecoder(hidden_channels)

        # Move model components to the selected device
        self.to(self.device)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        """
        Forward pass through the model.
        
        Parameters:
        - x_dict: Dictionary of node features
        - edge_index_dict: Dictionary of edge indices
        - edge_label_index: Edge indices to predict
        
        Returns:
        - scores: Edge prediction scores
        """
        # Decode and Encode
        return self.decoder(self.encoder(x_dict, edge_index_dict), edge_label_index)