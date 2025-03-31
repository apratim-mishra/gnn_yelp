import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.metrics import LinkPredMAP, LinkPredPrecision, LinkPredRecall
from torch_geometric.nn import MIPSKNNIndex
from tqdm import tqdm

def bpr_loss(positive_scores, negative_scores):
    """
    Computes the Bayesian Personalization Maximization (BPR) loss for a batch of positive and negative scores.
    """
    return -F.logsigmoid(positive_scores - negative_scores).mean()

def bce_loss(pred, target):
    """
    Computes the Binary Cross-Entropy (BCE) loss for a batch of predictions and targets.
    """
    return F.binary_cross_entropy_with_logits(pred, target.float())

def train_bpr(model, train_loader, optimizer, device):
    """
    Train function using BPR loss.
    
    Parameters:
    - model: PyTorch model
    - train_loader: PyTorch DataLoader for training data
    - optimizer: PyTorch optimizer
    - device: Computation device (CPU/GPU)
    
    Returns:
    - average_loss: Average loss for the epoch
    """
    model.train()  # Set model to training mode
    total_loss = total_examples = 0

    # Loop through each batch of training data
    for sampled_train_data in train_loader:
        sampled_train_data = sampled_train_data.to(device)
        optimizer.zero_grad()

        # Get edge indices for the current batch
        edge_label_index = sampled_train_data['user', 'reviews', 'restaurant'].edge_label_index.to(device)

        # Separate positive and negative edges
        pos_mask = sampled_train_data['user', 'reviews', 'restaurant'].edge_label == 1
        pos_edge_index = edge_label_index[:, pos_mask].to(device)
        neg_edge_index = edge_label_index[:, ~pos_mask].to(device)

        # Forward pass for both positive and negative samples
        pos_pred = model(sampled_train_data.x_dict, sampled_train_data.edge_index_dict, pos_edge_index)
        neg_pred = model(sampled_train_data.x_dict, sampled_train_data.edge_index_dict, neg_edge_index)

        # Compute BPR loss and backpropagate
        loss = bpr_loss(pos_pred, neg_pred)
        loss.backward()
        optimizer.step()

        # Accumulate total loss and examples
        total_loss += float(loss) * edge_label_index.numel()
        total_examples += edge_label_index.numel()

    # Return average loss per example
    return total_loss / total_examples

def train_bce(model, train_loader, optimizer, device):
    """
    Train function using BCE loss.
    
    Parameters:
    - model: PyTorch model
    - train_loader: PyTorch DataLoader for training data
    - optimizer: PyTorch optimizer
    - device: Computation device (CPU/GPU)
    
    Returns:
    - average_loss: Average loss for the epoch
    """
    model.train()  # Set model to training mode
    total_loss = total_examples = 0

    # Loop through each batch of training data
    for sampled_train_data in train_loader:
        sampled_train_data = sampled_train_data.to(device)
        optimizer.zero_grad()

        # Get actual edge labels and predicted values from the model
        actual = sampled_train_data['user', 'reviews', 'restaurant'].edge_label
        predicted = model(
            sampled_train_data.x_dict,
            sampled_train_data.edge_index_dict,
            sampled_train_data['user', 'reviews', 'restaurant'].edge_label_index,
        )

        # Compute BCE loss and backpropagate
        loss = bce_loss(predicted, actual)
        loss.backward()
        optimizer.step()

        # Accumulate total loss and examples
        total_loss += float(loss) * actual.numel()
        total_examples += actual.numel()

    # Return average loss per example
    return total_loss / total_examples

def train_bpr_with_hard_sampling(model, train_loader, optimizer, ref_model, device, num_hard_negatives=100):
    """
    Train using BPR loss with hard negative sampling.
    
    Parameters:
    - model: PyTorch model to train
    - train_loader: PyTorch DataLoader for training data
    - optimizer: PyTorch optimizer
    - ref_model: Reference model used to identify hard negatives
    - device: Computation device (CPU/GPU)
    - num_hard_negatives: Number of hard negative examples to use
    
    Returns:
    - average_loss: Average loss for the epoch
    """
    model.train()
    total_loss = total_examples = 0

    for sampled_train_data in tqdm(train_loader, desc="Training with hard sampling"):
        sampled_train_data = sampled_train_data.to(device)
        optimizer.zero_grad()

        edge_label_index = sampled_train_data['user', 'reviews', 'restaurant'].edge_label_index.to(device)
        pos_mask = sampled_train_data['user', 'reviews', 'restaurant'].edge_label == 1
        pos_edge_index = edge_label_index[:, pos_mask].to(device)
        neg_edge_index = edge_label_index[:, ~pos_mask].to(device)

        num_samples = pos_edge_index.size(1)
        if num_samples == 0:
            continue  # Skip this batch if there are no positive samples

        # Generate hard negative samples
        with torch.no_grad():
            # use previously trained model to get scores for negative edges
            neg_pred = ref_model(sampled_train_data.x_dict, sampled_train_data.edge_index_dict, neg_edge_index)
            # sort scores and select `num_hard_negatives` hardest edges
            neg_pred_idx = neg_pred.argsort(descending=True)[:num_hard_negatives] # indices of neg pred with highest score
            hard_neg_edge_index = neg_edge_index[:, neg_pred_idx]
            # combine with normal edges
            normal_neg_idx = torch.randint(0, num_samples * 3, (num_samples-num_hard_negatives, ), device=device)
            normal_neg_edge = neg_edge_index[:, normal_neg_idx]
            hard_neg_edge_index = torch.cat((hard_neg_edge_index, normal_neg_edge), dim=1)

        # Forward pass for positive and hard negative samples
        pos_pred = model(sampled_train_data.x_dict, sampled_train_data.edge_index_dict, pos_edge_index)
        neg_pred = model(sampled_train_data.x_dict, sampled_train_data.edge_index_dict, hard_neg_edge_index)

        # Compute BPR loss and backpropagate
        loss = bpr_loss(pos_pred, neg_pred)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * pos_edge_index.size(1)
        total_examples += pos_edge_index.size(1)

    return total_loss / total_examples if total_examples > 0 else 0.0

@torch.no_grad()
def test(model, test_data, k=5, device=None):
    """
    Evaluate model on test data.
    
    Parameters:
    - model: PyTorch model to evaluate
    - test_data: Test dataset
    - k: Number of top recommendations to consider
    - device: Computation device (CPU/GPU)
    
    Returns:
    - map_score: Mean Average Precision score
    - precision_score: Precision@k score
    - recall_score: Recall@k score
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    test_data = test_data.to(device)

    # Compute node embeddings
    emb = model.encoder(
        test_data.x_dict,
        test_data.edge_index_dict,
    )
    restaurant_emb = emb['restaurant']
    user_emb = emb['user']

    # Decode embeddings
    restaurant_embeddings = model.decoder.restaurant_lin(restaurant_emb)
    user_embeddings = model.decoder.user_lin(user_emb)

    # Edge label indices
    edge_label_index = test_data['user', 'reviews', 'restaurant'].edge_label_index

    # Instantiate k-NN index based on maximum inner product search (MIPS)
    mips = MIPSKNNIndex(restaurant_embeddings)

    # Initialize metrics
    map_metric = LinkPredMAP(k=k).to(device)
    precision_metric = LinkPredPrecision(k=k).to(device)
    recall_metric = LinkPredRecall(k=k).to(device)

    # Perform MIPS search:
    _, pred_index_mat = mips.search(user_embeddings, k)

    # Update retrieval metrics:
    map_metric.update(pred_index_mat, edge_label_index)
    precision_metric.update(pred_index_mat, edge_label_index)
    recall_metric.update(pred_index_mat, edge_label_index)

    return (
        float(map_metric.compute()),
        float(precision_metric.compute()),
        float(recall_metric.compute()),
    )

@torch.no_grad()
def roc_auc_test(model, test_data, device=None):
    """
    Calculate ROC-AUC score for model predictions.
    
    Parameters:
    - model: PyTorch model to evaluate
    - test_data: Test dataset
    - device: Computation device (CPU/GPU)
    
    Returns:
    - auc_score: Area Under the ROC Curve score
    """
    if device is None:
        device = next(model.parameters()).device
        
    model.eval()
    test_data = test_data.to(device)

    # Forward pass for test data
    test_edge_label_index = test_data['user', 'reviews', 'restaurant'].edge_label_index.to(device)
    test_target = test_data['user', 'reviews', 'restaurant'].edge_label.to(device)
    test_pred = model(test_data.x_dict, test_data.edge_index_dict, test_edge_label_index)
    all_preds = test_pred.cpu().numpy()
    all_targets = test_target.cpu().numpy()

    # Compute AUC score
    test_auc = roc_auc_score(all_targets, all_preds)
    return test_auc