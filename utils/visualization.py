import matplotlib.pyplot as plt

def plot_auc(bce_auc, bpr_auc, epochs, title="Validation AUC Values Over Epochs"):
    """
    Plot AUC scores over epochs for BCE and BPR models.
    
    Parameters:
    - bce_auc: List of AUC scores for BCE model
    - bpr_auc: List of AUC scores for BPR model
    - epochs: List of epoch numbers
    - title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, bce_auc, label='Validation AUC (BCE)', marker='o')
    plt.plot(epochs, bpr_auc, label='Validation AUC (BPR)', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_training_loss(bce_loss, bpr_loss, epochs, title="Training Loss Over Epochs"):
    """
    Plot training loss over epochs for BCE and BPR models.
    
    Parameters:
    - bce_loss: List of training losses for BCE model
    - bpr_loss: List of training losses for BPR model
    - epochs: List of epoch numbers
    - title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, bce_loss, label='BCE Training Loss', marker='o')
    plt.plot(epochs, bpr_loss, label='BPR Training Loss', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_recall(bce_recall, bpr_recall, epochs, k, title=None):
    """
    Plot recall scores over epochs for BCE and BPR models.
    
    Parameters:
    - bce_recall: List of recall scores for BCE model
    - bpr_recall: List of recall scores for BPR model
    - epochs: List of epoch numbers
    - k: Value of k for Recall@k
    - title: Plot title (optional)
    """
    if title is None:
        title = f'Validation Recall@{k} Values Over Epochs'
        
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, bce_recall, label='Validation Recall (BCE)', marker='o')
    plt.plot(epochs, bpr_recall, label='Validation Recall (BPR)', marker='^')
    plt.xlabel('Epochs')
    plt.ylabel(f'Recall@{k}')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_precision(bce_precision, bpr_precision, epochs, k, title=None):
    """
    Plot precision scores over epochs for BCE and BPR models.
    
    Parameters:
    - bce_precision: List of precision scores for BCE model
    - bpr_precision: List of precision scores for BPR model
    - epochs: List of epoch numbers
    - k: Value of k for Precision@k
    - title: Plot title (optional)
    """
    if title is None:
        title = f'Validation Precisions@{k} Values Over Epochs'
        
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, bce_precision, label='Validation Precisions (BCE)', marker='o')
    plt.plot(epochs, bpr_precision, label='Validation Precisions (BPR)', marker='^')
    plt.xlabel('Epochs')
    plt.ylabel(f'Precision@{k}')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def compare_all_models(model_results, metric_name, epochs, title=None):
    """
    Compare multiple models on a specific metric.
    
    Parameters:
    - model_results: Dictionary mapping model names to their metric scores
    - metric_name: Name of the metric being compared
    - epochs: List of epoch numbers
    - title: Plot title (optional)
    """
    if title is None:
        title = f'Comparison of {metric_name} Across Models'
        
    plt.figure(figsize=(12, 7))
    markers = ['o', 'x', 's', 'd', '^', 'v', '<', '>', 'p', '*']
    
    for idx, (model_name, results) in enumerate(model_results.items()):
        plt.plot(epochs, results, label=model_name, marker=markers[idx % len(markers)])
    
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()