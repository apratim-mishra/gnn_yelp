import os
import argparse
import torch
import numpy as np
from data_processor import process_yelp_data
from train import train_all_models, train_model
from utils.preprocessing import load_tensor
from utils.evaluation import test, roc_auc_test
from models.baseline import BaselineModel
from models.improved import ImprovedModel

def parse_args():
    parser = argparse.ArgumentParser(description='Yelp Restaurant Recommendations with GNNs')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing Yelp dataset files')
    parser.add_argument('--action', type=str, choices=['process', 'train', 'evaluate', 'all'], default='all',
                       help='Action to perform: process data, train models, evaluate models, or all')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--k', type=int, default=300, help='Value of k for Recall@k and Precision@k')
    parser.add_argument('--model_dir', type=str, default='trained_models', help='Directory to save trained models')
    parser.add_argument('--model_type', type=str, choices=['baseline', 'improved', 'hard_sampling', 'all'], 
                       default='all', help='Type of model to train or evaluate')
    parser.add_argument('--load_model', type=str, default=None, help='Path to model to load for evaluation')
    parser.add_argument('--seed', type=int, default=224, help='Random seed for reproducibility')
    parser.add_argument('--state', type=str, default='CA', help='Only include businesses from this state')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Check FAISS compatibility
    try:
        import faiss
        has_faiss_gpu = hasattr(faiss, 'index_cpu_to_gpu')
        if torch.cuda.is_available() and not has_faiss_gpu:
            print("WARNING: Using faiss-cpu, but GPU is available. Some operations will be slower.")
            print("Consider installing faiss-gpu for better performance.")
    except ImportError:
        print("WARNING: FAISS not installed. Some operations may fail.")
    
    # Set device based on FAISS compatibility
    use_gpu = torch.cuda.is_available()
    if use_gpu and not has_faiss_gpu:
        print("WARNING: Using CPU for evaluation due to faiss-cpu limitation")
        eval_device = torch.device('cpu')
    else:
        eval_device = torch.device('cuda' if use_gpu else 'cpu')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device for training: {device}')
    print(f'Using device for evaluation: {eval_device}')
    
    try:
        # Create output directories
        os.makedirs(args.model_dir, exist_ok=True)
        processed_dir = os.path.join(args.data_dir, 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        
        # Process data
        if args.action in ['process', 'all']:
            print("\n=== Processing Yelp Dataset ===")
            graph, _ = process_yelp_data(args.data_dir, state_filter=args.state)
        else:
            # Load processed graph
            graph_path = os.path.join(args.data_dir, 'processed', 'yelp_graph.pt')
            if not os.path.exists(graph_path):
                raise FileNotFoundError(f"Processed graph not found at {graph_path}. Run with --action process first.")
            
            print(f"Loading graph from {graph_path}...")
            graph = load_tensor(graph_path)
            
            # Print graph info
            print("Graph loaded successfully")
            print(f"Graph node types: {graph.node_types}")
            print(f"Graph edge types: {graph.edge_types}")
        
        # Train models
        if args.action in ['train', 'all']:
            if args.model_type == 'all':
                print("\n=== Training All Models ===")
                models, test_metrics = train_all_models(
                    graph, 
                    num_epochs=args.epochs, 
                    batch_size=args.batch_size, 
                    k=args.k, 
                    save_dir=args.model_dir
                )
            else:
                print(f"\n=== Training {args.model_type.capitalize()} Model ===")
                try:
                    model_type = 'improved' if args.model_type == 'hard_sampling' else args.model_type
                    loss_type = 'bce'
                    hard_sampling = args.model_type == 'hard_sampling'
                    
                    if hard_sampling:
                        print("\n=== Training Reference Improved Model for Hard Sampling ===")
                        ref_model, _, _ = train_model(
                            graph, 'improved', 'bpr', 
                            num_epochs=args.epochs, 
                            batch_size=args.batch_size,
                            k=args.k,
                            save_dir=args.model_dir,
                            device=device
                        )
                    else:
                        ref_model = None
                    
                    model, (train_data, val_data, test_data), results = train_model(
                        graph, 
                        model_type, 
                        loss_type, 
                        num_epochs=args.epochs, 
                        batch_size=args.batch_size,
                        k=args.k,
                        hard_sampling=hard_sampling,
                        ref_model=ref_model,
                        save_dir=args.model_dir,
                        device=device
                    )
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("WARNING: GPU out of memory. Trying with CPU...")
                        device = torch.device('cpu')
                        torch.cuda.empty_cache()
                        # Retry with CPU
                        model, (train_data, val_data, test_data), results = train_model(
                            graph, 
                            model_type, 
                            loss_type, 
                            num_epochs=args.epochs, 
                            batch_size=args.batch_size,
                            k=args.k,
                            hard_sampling=hard_sampling,
                            ref_model=ref_model,
                            save_dir=args.model_dir,
                            device=device
                        )
                    else:
                        raise e
                
                # Evaluate on test data
                print("\n=== Evaluating on Test Data ===")
                map_metric, precision_metric, recall_metric = test(model, test_data, k=args.k, device=eval_device)
                auc = roc_auc_test(model, test_data, device=eval_device)
                
                print(f"Test MAP@{args.k}: {map_metric:.4f}")
                print(f"Test Precision@{args.k}: {precision_metric:.4f}")
                print(f"Test Recall@{args.k}: {recall_metric:.4f}")
                print(f"Test AUC: {auc:.4f}")
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    print("\nCompleted!")

if __name__ == "__main__":
    main()