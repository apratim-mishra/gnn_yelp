import os
import torch
import pandas as pd
import argparse
from torch_geometric.data import HeteroData

# Assume load_tensor is available, or define a simple version if not in utils
# from utils.preprocessing import load_tensor
def load_tensor(filepath):
    """Loads a tensor from a .pt file."""
    # Set weights_only=False to allow loading complex objects like HeteroData
    return torch.load(filepath, weights_only=False)

def analyze_graph(data_dir):
    """Loads and analyzes the processed Yelp HeteroData graph."""

    graph_path = os.path.join(data_dir, 'processed', 'yelp_graph.pt')
    encoded_business_path = os.path.join(data_dir, 'processed', 'encoded_business_df.csv')

    if not os.path.exists(graph_path):
        print(f"Error: Processed graph file not found at {graph_path}")
        print("Please run the data_processor.py script first.")
        return

    if not os.path.exists(encoded_business_path):
        print(f"Error: Encoded business file not found at {encoded_business_path}")
        print("Needed for restaurant feature names. Please ensure it was saved by data_processor.py.")
        return

    print(f"Loading graph from {graph_path}...")
    graph: HeteroData = load_tensor(graph_path)

    print("\n--- Basic Graph Properties ---")
    print(f"Node Types: {graph.node_types}")
    print(f"Edge Types: {graph.edge_types}")
    print(f"Is Undirected: {graph.is_undirected()}")

    print("\n--- Node Counts ---")
    for node_type in graph.node_types:
        print(f"Number of '{node_type}' nodes: {graph[node_type].num_nodes}")

    print("\n--- Edge Counts ---")
    for edge_type in graph.edge_types:
        print(f"Number of '{edge_type}' edges: {graph[edge_type].num_edges}")

    # --- User Node Analysis ---
    print("\n--- User Node Analysis ---")
    user_features = graph['user'].x
    print(f"User feature shape: {user_features.shape}")
    print(f"User feature data type: {user_features.dtype}")

    user_feature_names = [
        'review_count', 'yelping_since_encoded', 'useful', 'funny', 'cool',
        'elite_count', 'friends_count', 'fans', 'average_stars',
        'compliment_hot', 'compliment_more', 'compliment_profile',
        'compliment_cute', 'compliment_list', 'compliment_note',
        'compliment_plain', 'compliment_cool', 'compliment_funny',
        'compliment_writer', 'compliment_photos'
    ]

    if user_features.shape[1] == len(user_feature_names):
        print("User Feature Statistics:")
        user_stats_df = pd.DataFrame({
            'Feature': user_feature_names,
            'Mean': user_features.mean(dim=0).tolist(),
            'Std': user_features.std(dim=0).tolist(),
            'Min': user_features.min(dim=0).values.tolist(),
            'Max': user_features.max(dim=0).values.tolist(),
        })
        print(user_stats_df.to_string())
    else:
        print(f"Warning: Number of user features ({user_features.shape[1]}) does not match expected number ({len(user_feature_names)}). Skipping detailed stats.")
        print(f"Overall Mean: {user_features.mean():.4f}, Std: {user_features.std():.4f}, Min: {user_features.min():.4f}, Max: {user_features.max():.4f}")

    # --- Restaurant Node Analysis ---
    print("\n--- Restaurant Node Analysis ---")
    restaurant_features = graph['restaurant'].x
    print(f"Restaurant feature shape: {restaurant_features.shape}")
    print(f"Restaurant feature data type: {restaurant_features.dtype}")

    # Load encoded business dataframe to get feature names
    encoded_business_df = pd.read_csv(encoded_business_path)
    # Drop the business_id column if it exists, as it's not a feature
    if 'business_id' in encoded_business_df.columns:
        restaurant_feature_names = encoded_business_df.drop(columns=['business_id']).columns.tolist()
    else:
        restaurant_feature_names = encoded_business_df.columns.tolist()


    if restaurant_features.shape[1] == len(restaurant_feature_names):
        print("Restaurant Feature Statistics:")
        # Calculate stats safely, handling potential NaNs introduced before saving
        restaurant_features_safe = torch.nan_to_num(restaurant_features, nan=0.0)
        restaurant_stats_df = pd.DataFrame({
            'Feature': restaurant_feature_names,
            'Mean': restaurant_features_safe.mean(dim=0).tolist(),
            'Std': restaurant_features_safe.std(dim=0).tolist(),
            'Min': restaurant_features_safe.min(dim=0).values.tolist(),
            'Max': restaurant_features_safe.max(dim=0).values.tolist(),
        })
        # Display only a subset of features for brevity if too many
        if len(restaurant_feature_names) > 30:
             print("Displaying stats for first/last 15 features (due to large number):")
             print(restaurant_stats_df.head(15).to_string())
             print("...")
             print(restaurant_stats_df.tail(15).to_string())
        else:
            print(restaurant_stats_df.to_string())

    else:
         print(f"Warning: Number of restaurant features ({restaurant_features.shape[1]}) does not match number of columns in encoded_business_df ({len(restaurant_feature_names)}). Skipping detailed stats.")
         print(f"Overall Mean: {restaurant_features_safe.mean():.4f}, Std: {restaurant_features_safe.std():.4f}, Min: {restaurant_features_safe.min():.4f}, Max: {restaurant_features_safe.max():.4f}")


    # --- Review Edge Analysis ---
    print("\n--- Review Edge Analysis ---")
    # Note: The graph is made undirected, so check both edge types if they exist
    review_edge_types = [et for et in graph.edge_types if 'reviews' in et[1]]

    for edge_type in review_edge_types:
        print(f"\nAnalyzing Edge Type: {edge_type}")
        edge_index = graph[edge_type].edge_index
        edge_attr = graph[edge_type].edge_attr

        print(f"Edge index shape: {edge_index.shape}")
        print(f"Edge index data type: {edge_index.dtype}")
        print(f"Edge attribute shape: {edge_attr.shape}")
        print(f"Edge attribute data type: {edge_attr.dtype}")

        # Assuming original ('user', 'reviews', 'restaurant') edge attributes
        if edge_type == ('user', 'reviews', 'restaurant') or len(review_edge_types) == 1:
             edge_attr_names = ['stars', 'useful', 'funny', 'cool', 'date_encoded']
             if edge_attr.shape[1] == len(edge_attr_names):
                 print("Edge Attribute Statistics:")
                 edge_stats_df = pd.DataFrame({
                     'Attribute': edge_attr_names,
                     'Mean': edge_attr.mean(dim=0).tolist(),
                     'Std': edge_attr.std(dim=0).tolist(),
                     'Min': edge_attr.min(dim=0).values.tolist(),
                     'Max': edge_attr.max(dim=0).values.tolist(),
                 })
                 print(edge_stats_df.to_string())
             else:
                 print(f"Warning: Number of edge attributes ({edge_attr.shape[1]}) does not match expected number ({len(edge_attr_names)}). Skipping detailed stats.")
                 print(f"Overall Mean: {edge_attr.mean():.4f}, Std: {edge_attr.std():.4f}, Min: {edge_attr.min():.4f}, Max: {edge_attr.max():.4f}")
        elif edge_type == ('restaurant', 'rev_reviews', 'user'):
            print("Attributes for reverse edge type are typically the same as the forward edge type.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze the processed Yelp HeteroData graph')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the processed graph and associated files')
    args = parser.parse_args()

    analyze_graph(args.data_dir)
    print("\nGraph analysis complete.")