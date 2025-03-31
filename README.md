# Yelp Restaurant Recommendations with GNNs

A Graph Neural Network (GNN) based recommendation system for restaurants using the Yelp dataset. This project leverages heterogeneous graph neural networks to model user-restaurant interactions and make personalized restaurant recommendations.

## Project Overview

This system makes restaurant recommendations based on:
- User preferences and historical interactions
- Restaurant features (location, cuisine type, ratings, etc.)
- Graph structure of user-restaurant interactions

The project implements and compares several recommendation models:
1. **Baseline Model**: GraphSAGE-based GNN with simple edge prediction
2. **Improved Model**: Enhanced GNN with GAT, Transformer layers, and skip connections
3. **Hard Sampling Model**: Improved model with hard negative sampling

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric 2.6+
- CUDA-compatible GPU (recommended for faster training)

## Setup Instructions

### 1. Create a Python Environment

Using `uv` (a fast Python package installer):

```bash
# Install uv if not already installed 
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment
uv venv

# Activate the environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Alternatively, using traditional venv:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Using uv
uv pip install -r requirements_latest.txt

# Or using pip
pip install -r requirements_latest.txt
```

### 3. Prepare Data Directory

Create a directory for the Yelp dataset and download the JSON files from [Yelp Dataset](https://www.yelp.com/dataset):

```bash
mkdir -p data
# Place the following files in the data directory:
# - yelp_academic_dataset_business.json
# - yelp_academic_dataset_review.json
# - yelp_academic_dataset_user.json
# - yelp_academic_dataset_checkin.json
# - yelp_academic_dataset_tip.json
```

## Usage

### Data Processing

```bash
python download_nltk.py
```

Process the raw Yelp data to create a heterogeneous graph:

```bash
python data_processor.py --data_dir data
```

### Training Models

Train all recommendation models:

```bash
python train.py --data_dir data --epochs 100 --batch_size 512 --k 300 --model_dir trained_models
```

To train a specific model:

```bash
python train.py --data_dir data --model_type improved --loss_type bpr
```

### Running the Complete Pipeline

Process data, train models, and evaluate in one go:

```bash
python main.py --data_dir data --action all --epochs 100 --batch_size 512 --k 300
```

### Evaluating a Trained Model

```bash
python main.py --data_dir data --action evaluate --load_model trained_models/improved_bpr_model.pt --k 300
```

## Project Structure

```
yelp_recommendations/
├── data/                  # Data directory
├── models/                # Model implementations
│   ├── __init__.py
│   ├── baseline.py        # Baseline GraphSAGE model
│   ├── improved.py        # Improved model
├── utils/                 # Utility functions
│   ├── __init__.py
│   ├── preprocessing.py   # Data preprocessing functions
│   ├── evaluation.py      # Model evaluation functions
│   └── visualization.py   # Plotting functions
├── main.py                # Main script to run the pipeline
├── data_processor.py      # Data processing script
├── train.py               # Training script
├── requirements.txt       # Project dependencies
├── .gitignore             # Git ignore file
└── README.md              # Project documentation
```

## Model Details

### Baseline Model
- Two GraphSAGE convolution layers
- Simple dot product decoder for edge prediction

### Improved Model
- Enhanced architecture with GraphSAGE, GAT, and Transformer layers
- Batch normalization and skip connections
- Deeper MLP layers in the decoder with dropout

### Hard Sampling Model
- Based on the improved model
- Uses a trained model to identify "hard" negative examples
- Focuses training on more challenging cases to improve generalization

## Evaluation Metrics

The models are evaluated using:
- **Recall@K**: The proportion of relevant items found in the top K recommendations
- **Precision@K**: The proportion of relevant items among the top K recommendations
- **Mean Average Precision (MAP)**: The mean of the average precision scores for each user
- **Area Under the ROC Curve (AUC)**: Measures the model's ability to distinguish between positive and negative examples

## Loss Functions

Two loss functions are implemented and compared:
- **Bayesian Personalized Ranking (BPR)**: Optimizes the ranking of positive items relative to negative items
- **Binary Cross-Entropy (BCE)**: Treats recommendation as a binary classification problem


## License

This project is available under the MIT License.
