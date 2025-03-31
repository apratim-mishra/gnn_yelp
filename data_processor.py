import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.preprocessing import (
    download_nltk_resources, get_vader_sentiment, hours_to_total_hours_per_week,
    encode_features_for_businesses, save_tensor, load_tensor, encode_time
)
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

def process_yelp_data(data_dir, state_filter="CA", save_processed=True):
    """
    Process Yelp dataset files and create a heterogeneous graph.
    
    Parameters:
    - data_dir: Directory containing the Yelp dataset files
    - state_filter: Only include businesses from this state
    - save_processed: Whether to save processed data to disk
    
    Returns:
    - graph: A HeteroData object containing the processed graph
    - category_terms_list: List of restaurant categories
    """
    print("Downloading NLTK resources...")
    download_nltk_resources()
    
    # File paths
    business_filepath = os.path.join(data_dir, 'yelp_academic_dataset_business.json')
    review_filepath = os.path.join(data_dir, 'yelp_academic_dataset_review.json')
    checkin_filepath = os.path.join(data_dir, 'yelp_academic_dataset_checkin.json')
    tip_filepath = os.path.join(data_dir, 'yelp_academic_dataset_tip.json')
    user_filepath = os.path.join(data_dir, 'yelp_academic_dataset_user.json')
    
    # Create output directory if it doesn't exist
    if save_processed and not os.path.exists(os.path.join(data_dir, 'processed')):
        os.makedirs(os.path.join(data_dir, 'processed'))

    # Load data
    print("Loading business data...")
    business = pd.read_json(business_filepath, lines=True)
    
    print("Loading checkin data...")
    checkin = pd.read_json(checkin_filepath, lines=True)
    
    print("Loading tip data...")
    tip = pd.read_json(tip_filepath, lines=True)
    
    print("Processing businesses...")
    # Filter to get non-null and food/restaurant businesses
    business_no_null = business[~business['categories'].isnull()]
    business_restaurant = business_no_null[
        (business_no_null['categories'].str.contains('Restaurant').values) & 
        (business_no_null['state'] == state_filter)
    ]
    
    # Add total hours per week
    business_restaurant['total_hours_per_week'] = business_restaurant['hours'].apply(hours_to_total_hours_per_week)
    
    # Filter checkin by business id to obtain restaurants
    print("Processing checkins...")
    checkin_restaurants = checkin[checkin['business_id'].isin(business_restaurant['business_id'])]
    
    # Find counts for checkins
    checkin_restaurants['num_checkins'] = checkin_restaurants['date'].str.split(', ').apply(len)
    
    # Merge checkin information with business_restaurant
    business_and_checkins = business_restaurant.merge(
        checkin_restaurants[['business_id', 'num_checkins']], 
        on='business_id', 
        how='left'
    )
    business_and_checkins['num_checkins'] = business_and_checkins['num_checkins'].fillna(0)
    
    # Process tips
    print("Processing tips...")
    tip_restaurants = tip[tip['business_id'].isin(business_restaurant['business_id'])]
    
    # Use text and compute sentiment
    tip_restaurants['text'].fillna("", inplace=True)
    tip_restaurants['tip_text_sentiment'] = tip_restaurants['text'].apply(get_vader_sentiment)
    
    # Extract sentiment components
    tip_restaurants['neg_sentiment'] = tip_restaurants['tip_text_sentiment'].apply(lambda x: x[0])
    tip_restaurants['neu_sentiment'] = tip_restaurants['tip_text_sentiment'].apply(lambda x: x[1])
    tip_restaurants['pos_sentiment'] = tip_restaurants['tip_text_sentiment'].apply(lambda x: x[2])
    tip_restaurants['compound_sentiment'] = tip_restaurants['tip_text_sentiment'].apply(lambda x: x[3])
    
    # Aggregate sentiment values
    tip_restaurants['mean_neg_sentiment'] = tip_restaurants.groupby('business_id')['neg_sentiment'].transform('mean')
    tip_restaurants['mean_neu_sentiment'] = tip_restaurants.groupby('business_id')['neu_sentiment'].transform('mean')
    tip_restaurants['mean_pos_sentiment'] = tip_restaurants.groupby('business_id')['pos_sentiment'].transform('mean')
    tip_restaurants['mean_compound_sentiment'] = tip_restaurants.groupby('business_id')['compound_sentiment'].transform('mean')
    
    # Aggregate tips
    tip_restaurants['compliment_count'] = tip_restaurants.groupby('business_id')['compliment_count'].transform('sum')
    tip_restaurants_agg = tip_restaurants[
        ['business_id', 'mean_compound_sentiment', 'mean_neg_sentiment', 
         'mean_neu_sentiment', 'mean_pos_sentiment', 'compliment_count']
    ].drop_duplicates()
    
    # Merge tip data with business data
    business_and_checkins_and_tips = business_and_checkins.merge(
        tip_restaurants_agg, 
        on='business_id', 
        how='left'
    )
    
    # Extract category terms
    all_categories = ', '.join(business_restaurant['categories'].dropna())
    unique_categories = list(set([cat.strip() for cat in all_categories.split(',')]))
    
    # Create a list of common restaurant categories
    restaurant_categories = [
        cat for cat in unique_categories 
        if any(term in cat.lower() for term in [
            'restaurant', 'food', 'cuisine', 'bar', 'pub', 'cafe', 'coffee', 
            'bakery', 'tea', 'dessert', 'dining', 'grill', 'bistro'
        ])
    ]
    
    # Save category list
    category_terms_list = sorted(restaurant_categories)
    category_terms_df = pd.DataFrame(category_terms_list)
    if save_processed:
        category_terms_df.to_csv(os.path.join(data_dir, 'processed', 'category_terms.csv'), index=False)
    
    # Encode business features
    print("Encoding business features...")
    encoded_business_df = encode_features_for_businesses(business_and_checkins_and_tips, category_terms_list)
    encoded_business_df = encoded_business_df.fillna(0)
    
    if save_processed:
        encoded_business_df.to_csv(os.path.join(data_dir, 'processed', 'encoded_business_df.csv'), index=False)
        business_and_checkins_and_tips.to_csv(os.path.join(data_dir, 'processed', 'business_full_merged.csv'), index=False)
    
    # Create mapping from business IDs to indices
    valid_business_ids = set(encoded_business_df['business_id'].to_list())
    business_id_to_index = {
        business_id: index for index, business_id in enumerate(encoded_business_df['business_id'].to_list())
    }
    
    # Process reviews
    print("Processing reviews...")
    reviews_idx_processed = os.path.join(data_dir, 'processed', 'reviews_index.pt')
    reviews_attr_processed = os.path.join(data_dir, 'processed', 'reviews_attr.pt')
    
    if save_processed and os.path.isfile(reviews_idx_processed) and os.path.isfile(reviews_attr_processed):
        reviews_index = load_tensor(reviews_idx_processed)
        reviews_attr = load_tensor(reviews_attr_processed)
        
        # Extract valid user IDs from reviews
        valid_user_ids = set([source_node_user_id for source_node_user_id, _ in reviews_index])
    else:
        # Initialize empty lists for reviews index and attributes
        reviews_index = []
        reviews_attr = []
        valid_user_ids = []
        
        # Open and read the reviews.json file to process reviews data
        with open(review_filepath) as input_file:
            for line in tqdm(input_file, desc="Processing reviews"):
                # Parse each line (review data) into a dictionary
                review_dict = json.loads(line)
                source_node_user_id, dest_node_business_id = review_dict['user_id'], review_dict['business_id']
                
                # Only process reviews for valid businesses
                if dest_node_business_id in valid_business_ids:
                    # Append user-business pair (index) to reviews_index
                    reviews_index.append([source_node_user_id, dest_node_business_id])
                    
                    # Append review attributes to reviews_attr
                    reviews_attr.append([
                        review_dict['stars'],
                        review_dict['useful'],
                        review_dict['funny'],
                        review_dict['cool'],
                        encode_time(review_dict['date']),
                    ])
                    
                    # Keep track of valid user IDs for further processing
                    valid_user_ids.append(source_node_user_id)
        
        if save_processed:
            # Save the processed review index and attributes to disk
            save_tensor(reviews_idx_processed, reviews_index)
            save_tensor(reviews_attr_processed, reviews_attr)
        
    # Convert the list of valid user IDs to a set for uniqueness
    valid_user_ids = set(valid_user_ids)
    
    # Create a mapping of user IDs to a continuous index
    user_id_to_index = {user_id: index for index, user_id in enumerate(valid_user_ids)}
    
    # Convert the review index into a tensor of user indices and business indices
    reviews_index_tensor = torch.tensor([
        [user_id_to_index[source_node_user_id], business_id_to_index[dest_node_business_id]]
        for source_node_user_id, dest_node_business_id in reviews_index
    ]).T  # Transpose to get user_idx as the first row and business_idx as the second row
    
    # Convert the review attributes to a tensor
    reviews_attr_tensor = torch.tensor(reviews_attr)
    
    # Process users
    print("Processing users...")
    users_x_processed = os.path.join(data_dir, 'processed', 'user_processed.pt')
    
    if save_processed and os.path.isfile(users_x_processed):
        user_x = load_tensor(users_x_processed)
    else:
        # Open and read the user data from the input file
        with open(user_filepath) as input_file:
            # Initialize a tensor with zeros to store user feature data
            user_x = torch.zeros((len(valid_user_ids), 20))
            
            # Iterate through each line in the user data file
            for line in tqdm(input_file, desc="Processing users"):
                # Parse the line into a dictionary containing user information
                user_dict = json.loads(line)
                user_id = user_dict['user_id']
                
                # Only process the data for users that are in the set of valid user IDs
                if user_id in valid_user_ids:
                    # Calculate elite status count
                    elite_count = 0
                    if user_dict['elite']:
                        elite_count = user_dict['elite'].count(',') + 1 if ',' in user_dict['elite'] else 1
                        
                    # Populate the feature tensor with the user's data
                    user_x[user_id_to_index[user_id]] = torch.tensor([
                        user_dict['review_count'],
                        encode_time(user_dict['yelping_since']),
                        user_dict['useful'],
                        user_dict['funny'],
                        user_dict['cool'],
                        elite_count,
                        len(user_dict['friends'].split(',')) if user_dict['friends'] else 0,
                        user_dict['fans'],
                        user_dict['average_stars'],
                        user_dict['compliment_hot'],
                        user_dict['compliment_more'],
                        user_dict['compliment_profile'],
                        user_dict['compliment_cute'],
                        user_dict['compliment_list'],
                        user_dict['compliment_note'],
                        user_dict['compliment_plain'],
                        user_dict['compliment_cool'],
                        user_dict['compliment_funny'],
                        user_dict['compliment_writer'],
                        user_dict['compliment_photos'],
                    ])
        
        if save_processed:
            # Save the populated user features tensor to disk for future use
            save_tensor(users_x_processed, user_x)
    
    # Create the heterogeneous graph
    print("Creating heterogeneous graph...")
    
    # Extract restaurant node features (drop business_id column)
    restaurant_node_df = encoded_business_df.drop(columns=['business_id'])
    restaurant_node_x = torch.tensor(restaurant_node_df.values, dtype=torch.float32)
    
    # Create the graph
    yelp_graph = HeteroData()
    yelp_graph['restaurant'].x = restaurant_node_x
    yelp_graph['user'].x = user_x.float()  # Ensure float type
    yelp_graph['user', 'reviews', 'restaurant'].edge_index = reviews_index_tensor
    yelp_graph['user', 'reviews', 'restaurant'].edge_attr = reviews_attr_tensor.float()
    
    # Add reverse edges for better message passing
    yelp_graph = ToUndirected()(yelp_graph)
    
    # Ensure no NaN values
    yelp_graph['restaurant'].x = torch.nan_to_num(yelp_graph['restaurant'].x, nan=0.0)
    
    if save_processed:
        save_tensor(os.path.join(data_dir, 'processed', 'yelp_graph.pt'), yelp_graph)
    
    return yelp_graph, category_terms_list

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Yelp dataset for restaurant recommendations')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing Yelp dataset files')
    parser.add_argument('--state', type=str, default='CA', help='Only include businesses from this state')
    args = parser.parse_args()
    
    process_yelp_data(args.data_dir, state_filter=args.state)
    print("Data processing completed successfully!")