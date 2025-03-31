import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader

from utils.preprocessing import load_tensor
from utils.evaluation import train_bpr, train_bce, train_bpr_with_hard_sampling, test, roc_auc_test
from utils.visualization import plot_auc, plot_training_loss, plot_recall, plot_precision, compare_all_models
from models.baseline import BaselineModel
from models.improved import ImprovedModel

def train_model(graph, model_type, loss_type, num_epochs=100, batch_size=512, hidden_dim=32, learning_rate=0.01, 
                k=300, seed=224, val_ratio=0.15, test_ratio=0.15, hard_sampling=False, ref_model=None,
                save_dir=None, device=None, eval_device=None):
    """
    Train a recommendation model.
    
    Parameters:
    - graph: HeteroData graph object
    - model_type: Type of model ('baseline' or 'improved')
    - loss_type: Type of loss function ('bpr' or 'bce')
    - num_epochs: Number of training epochs
    - batch_size: Batch size for training
    - hidden_dim: Number of hidden dimensions in the model
    - learning_rate: Learning rate for optimizer
    - k: Value of k for Recall@k and Precision@k
    - seed: Random seed for reproducibility
    - val_ratio: Ratio of validation data
    - test_ratio: Ratio of test data
    - hard_sampling: Whether to use hard negative sampling
    - ref_model: Reference model for hard negative sampling
    - save_dir: Directory to save model and results
    - device: Computation device for training (CPU/GPU)
    - eval_device: Computation device for evaluation (CPU/GPU)
    
    Returns:
    - model: Trained model
    - train_data, val_data, test_data: Data splits
    - results: Dictionary containing training metrics
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if eval_device is None:
        eval_device = device
        
    print(f'Using device for training: {device}')
    print(f'Using device for evaluation: {eval_device}')
    
    # Memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("Starting data split...")
    try:
        # Split the graph for training, validation, and test
        transform = RandomLinkSplit(
            num_val=val_ratio,
            num_test=test_ratio,
            disjoint_train_ratio=0.3,
            neg_sampling_ratio=0.5,  # Reduced from 1.0
            add_negative_train_samples=False,
            edge_types=[('user', 'reviews', 'restaurant')],
            rev_edge_types=[('restaurant', 'rev_reviews', 'user')]
        )
        
        # Verify edge types exist in the graph
        if ('user', 'reviews', 'restaurant') not in graph.edge_types:
            print(f"WARNING: Edge type ('user', 'reviews', 'restaurant') not found in graph!")
            print(f"Available edge types: {graph.edge_types}")
            raise ValueError("Edge type not found in graph")
            
        train_data, val_data, test_data = transform(graph)
        print("Data split successful!")
    except Exception as e:
        print(f"Error during data split: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    try:
        print("Creating data loader...")
        # Create data loader
        edge_label_index = train_data["user", "reviews", "restaurant"].edge_label_index
        edge_label = train_data["user", "reviews", "restaurant"].edge_label
        
        # For BPR loss, we need a 1:1 ratio of positive to negative samples
        neg_ratio = 1.0 if loss_type == 'bpr' else 0.2
        
        train_loader = LinkNeighborLoader(
            data=train_data,
            num_neighbors=[5, 2],  # Reduced from [20, 10]
            neg_sampling_ratio=neg_ratio,  # Use 1.0 for BPR loss
            edge_label_index=(('user', 'reviews', 'restaurant'), edge_label_index),
            edge_label=edge_label,
            batch_size=batch_size,
            shuffle=True,
        )
        print("Data loader created successfully!")
    except Exception as e:
        print(f"Error creating data loader: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Initialize model
    print(f"Initializing {model_type} model...")
    try:
        if model_type == 'baseline':
            model = BaselineModel(graph.metadata(), hidden_channels=hidden_dim)
        elif model_type == 'improved':
            model = ImprovedModel(graph.metadata(), hidden_channels=hidden_dim)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        print("Model initialized successfully!")
    except Exception as e:
        print(f"Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Training metrics
    training_losses = []
    val_maps = []
    val_precisions = []
    val_recalls = []
    val_auc_values = []
    epochs = list(range(num_epochs))
    
    # Training loop
    print(f"Starting training loop for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        try:
            # Train step
            model.train()
            if loss_type == 'bpr':
                if hard_sampling and ref_model is not None:
                    train_loss = train_bpr_with_hard_sampling(model, train_loader, optimizer, ref_model, device)
                else:
                    train_loss = train_bpr(model, train_loader, optimizer, device)
            elif loss_type == 'bce':
                train_loss = train_bce(model, train_loader, optimizer, device)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            
            training_losses.append(train_loss)
            
            # Validation step
            model.eval()
            val_map_metric, val_precision_metric, val_recall_metric = test(model, val_data, k=k, device=eval_device)
            val_maps.append(val_map_metric)
            val_precisions.append(val_precision_metric)
            val_recalls.append(val_recall_metric)
            
            # AUC evaluation
            auc = roc_auc_test(model, val_data, device=eval_device)
            val_auc_values.append(auc)
            
            # Print progress
            print(f"\nEpoch: {epoch}, Training Loss: {train_loss:.4f}")
            print(f"Val MAP@{k}: {val_map_metric:.4f}, Val Precision@{k}: {val_precision_metric:.4f}, "
                  f"Val Recall@{k}: {val_recall_metric:.4f}, Val AUC: {auc:.4f}")
            
            # Clear GPU cache after each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error during epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
            if epoch > 0:  # If we have some results, continue with what we have
                break
            else:
                raise
    
    # Save model if requested
    if save_dir is not None and len(training_losses) > 0:
        os.makedirs(save_dir, exist_ok=True)
        model_name = f"{model_type}_{loss_type}_model"
        if hard_sampling:
            model_name += "_hard_sampling"
        torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}.pt"))
        
        # Save training metrics
        results = {
            'epochs': epochs[:len(training_losses)],
            'training_losses': training_losses,
            'val_maps': val_maps,
            'val_precisions': val_precisions,
            'val_recalls': val_recalls,
            'val_auc_values': val_auc_values
        }
        torch.save(results, os.path.join(save_dir, f"{model_name}_results.pt"))
    
    # Return model and metrics
    results = {
        'training_losses': training_losses,
        'val_maps': val_maps,
        'val_precisions': val_precisions,
        'val_recalls': val_recalls,
        'val_auc_values': val_auc_values
    }
    
    return model, (train_data, val_data, test_data), results

def train_all_models(graph, num_epochs=100, batch_size=512, k=300, save_dir=None):
    """
    Train all model variants and compare their performance.
    
    Parameters:
    - graph: HeteroData graph object
    - num_epochs: Number of training epochs
    - batch_size: Batch size for training
    - k: Value of k for Recall@k and Precision@k
    - save_dir: Directory to save models and results
    
    Returns:
    - models: Dictionary of trained models
    - results: Dictionary of model results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Store models and results
    models = {}
    data_splits = {}
    all_results = {}
    
    # Train baseline models
    print("\n=== Training Baseline Model with BCE Loss ===")
    models['baseline_bce'], data_splits['baseline_bce'], all_results['baseline_bce'] = train_model(
        graph, 'baseline', 'bce', num_epochs=num_epochs, batch_size=batch_size, 
        hidden_dim=32, k=k, save_dir=save_dir, device=device
    )
    
    print("\n=== Training Baseline Model with BPR Loss ===")
    models['baseline_bpr'], data_splits['baseline_bpr'], all_results['baseline_bpr'] = train_model(
        graph, 'baseline', 'bpr', num_epochs=num_epochs, batch_size=batch_size, 
        hidden_dim=32, k=k, save_dir=save_dir, device=device
    )
    
    # Train improved models
    print("\n=== Training Improved Model with BCE Loss ===")
    models['improved_bce'], data_splits['improved_bce'], all_results['improved_bce'] = train_model(
        graph, 'improved', 'bce', num_epochs=num_epochs, batch_size=batch_size, 
        hidden_dim=64, k=k, save_dir=save_dir, device=device
    )
    
    print("\n=== Training Improved Model with BPR Loss ===")
    models['improved_bpr'], data_splits['improved_bpr'], all_results['improved_bpr'] = train_model(
        graph, 'improved', 'bpr', num_epochs=num_epochs, batch_size=batch_size, 
        hidden_dim=64, k=k, save_dir=save_dir, device=device
    )
    
    # Train improved model with hard negative sampling
    print("\n=== Training Improved Model with Hard Negative Sampling ===")
    models['improved_hard'], data_splits['improved_hard'], all_results['improved_hard'] = train_model(
        graph, 'improved', 'bpr', num_epochs=num_epochs, batch_size=batch_size, 
        hidden_dim=64, k=k, hard_sampling=True, ref_model=models['improved_bpr'],
        save_dir=save_dir, device=device
    )
    
    # Compare models on test data
    test_metrics = {}
    print("\n=== Evaluating All Models on Test Data ===")
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        _, _, test_data = data_splits[model_name]
        map_metric, precision_metric, recall_metric = test(model, test_data, k=k, device=device)
        auc = roc_auc_test(model, test_data, device=device)
        
        test_metrics[model_name] = {
            'map': map_metric,
            'precision': precision_metric,
            'recall': recall_metric,
            'auc': auc
        }
        
        print(f"Test MAP@{k}: {map_metric:.4f}")
        print(f"Test Precision@{k}: {precision_metric:.4f}")
        print(f"Test Recall@{k}: {recall_metric:.4f}")
        print(f"Test AUC: {auc:.4f}")
    
    # Save test metrics
    if save_dir is not None:
        torch.save(test_metrics, os.path.join(save_dir, "test_metrics.pt"))
    
    # Compare models visually
    epochs = list(range(num_epochs))
    
    # AUC comparison
    compare_all_models(
        {name: results['val_auc_values'] for name, results in all_results.items()},
        'AUC', epochs, 'Validation AUC Comparison Across Models'
    )
    
    # Loss comparison
    compare_all_models(
        {name: results['training_losses'] for name, results in all_results.items()},
        'Training Loss', epochs, 'Training Loss Comparison Across Models'
    )
    
    # Recall comparison
    compare_all_models(
        {name: results['val_recalls'] for name, results in all_results.items()},
        f'Recall@{k}', epochs, f'Validation Recall@{k} Comparison Across Models'
    )
    
    # Precision comparison
    compare_all_models(
        {name: results['val_precisions'] for name, results in all_results.items()},
        f'Precision@{k}', epochs, f'Validation Precision@{k} Comparison Across Models'
    )
    
    return models, test_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train recommendation models on Yelp dataset')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing processed Yelp data')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--k', type=int, default=300, help='Value of k for Recall@k and Precision@k')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save trained models')
    parser.add_argument('--model_type', type=str, choices=['baseline', 'improved', 'all'], default='all',
                        help='Type of model to train (baseline, improved, or all)')
    parser.add_argument('--loss_type', type=str, choices=['bce', 'bpr', 'both'], default='both',
                        help='Type of loss function (bce, bpr, or both)')
    args = parser.parse_args()
    
    # Load processed graph
    graph_path = os.path.join(args.data_dir, 'processed', 'yelp_graph.pt')
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Processed graph not found at {graph_path}. Run data_processor.py first.")
    
    print(f"Loading graph from {graph_path}...")
    graph = load_tensor(graph_path)
    
    if args.model_type == 'all':
        # Train all models
        train_all_models(graph, num_epochs=args.epochs, batch_size=args.batch_size, k=args.k, save_dir=args.model_dir)
    else:
        # Train specific model type and loss function
        loss_types = ['bce', 'bpr'] if args.loss_type == 'both' else [args.loss_type]
        
        for loss_type in loss_types:
            print(f"\n=== Training {args.model_type.capitalize()} Model with {loss_type.upper()} Loss ===")
            train_model(
                graph, args.model_type, loss_type, 
                num_epochs=args.epochs, 
                batch_size=args.batch_size,
                k=args.k,
                save_dir=args.model_dir
            )
    
    print("Training completed!")