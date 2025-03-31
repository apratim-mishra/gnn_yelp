import torch
import torch.nn.functional as F
from torch.nn import LayerNorm, Dropout, Linear, ReLU, Sequential, BatchNorm1d
from torch_geometric.nn import SAGEConv, GATConv, to_hetero, TransformerConv, GCNConv

class GNNEncoderImproved(torch.nn.Module):
    """
    Improved GNN encoder using a combination of GraphSAGE, GAT, and Transformer convolution layers.
    Includes batch normalization, skip connections, and dropout for better performance.
    """
    def __init__(self, hidden_channels, out_channels, num_layers=2, dropout=0.2):
        super().__init__()

        # First Block - SAGE, BN, GatConv, Linear
        self.conv1 = SAGEConv((-1, -1, -1, -1), hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)
        self.gat1 = GATConv(hidden_channels, hidden_channels, heads=8, dropout=dropout, add_self_loops=False)
        self.skip1 = Linear(hidden_channels * 8, hidden_channels)

        # Second Block - SAGE, BN, GATConv, Linear
        self.conv2 = SAGEConv((-1, -1, -1, -1), hidden_channels)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.gat2 = GATConv(hidden_channels, hidden_channels, heads=8, dropout=dropout, add_self_loops=False)
        self.skip2 = Linear(hidden_channels * 8, hidden_channels)

        # Third Block - SAGE, Transformer, Linear
        self.conv3 = SAGEConv((-1, -1, -1, -1), hidden_channels)
        self.transformer_layer = TransformerConv(
            hidden_channels, hidden_channels, heads=4, dropout=dropout
        )
        self.skip3 = Linear(hidden_channels * 4, hidden_channels)

        # Output Block - SAGE, SAGE
        self.conv4 = SAGEConv((-1, -1, -1, -1), hidden_channels)
        self.conv5 = SAGEConv((-1, -1, -1, -1), out_channels)

    def forward(self, x, edge_index):
        # Apply first block
        x = self.conv1(x, edge_index).relu()
        x = self.bn1(x).relu()
        x = self.gat1(x, edge_index).relu()
        x = self.skip1(x).relu()

        # Apply second block
        x = self.conv2(x, edge_index).relu()
        x = self.bn2(x).relu()
        x = self.gat2(x, edge_index).relu()
        x = self.skip2(x).relu()

        # Apply third block
        x = self.conv3(x, edge_index).relu()
        x = self.transformer_layer(x, edge_index).relu()
        x = self.skip3(x).relu()

        # Apply output block
        x = self.conv4(x, edge_index).relu()
        x = self.conv5(x, edge_index)

        return x

class EdgeDecoderImproved(torch.nn.Module):
    """
    Improved edge decoder with deeper MLP layers and dropout.
    """
    def __init__(self, hidden_channels):
        super().__init__()
        self.user_lin = Sequential(
            Linear(hidden_channels, hidden_channels * 2),
            ReLU(),
            Dropout(0.2),
            Linear(hidden_channels * 2, hidden_channels)
        )

        self.restaurant_lin = Sequential(
            Linear(hidden_channels, hidden_channels * 2),
            ReLU(),
            Dropout(0.2),
            Linear(hidden_channels * 2, hidden_channels)
        )

    def forward(self, z_dict, edge_label_index):
        # Separate user and restaurant indices
        row, col = edge_label_index
        user_emb = z_dict['user'][row]
        restaurant_emb = z_dict['restaurant'][col]
        return (self.user_lin(user_emb) * self.restaurant_lin(restaurant_emb)).sum(dim=-1)

class ImprovedModel(torch.nn.Module):
    """
    Improved recommendation model with enhanced GNN architecture.
    """
    def __init__(self, metadata, hidden_channels=64, device=None):
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
        self.encoder = GNNEncoderImproved(hidden_channels, hidden_channels)
        # Convert the encoder to handle heterogeneous graphs
        self.encoder = to_hetero(self.encoder, metadata, aggr='sum')
        # Initialize the edge decoder
        self.decoder = EdgeDecoderImproved(hidden_channels)

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