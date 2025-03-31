import os
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import gender_guesser.detector as gender

# Check and download necessary NLTK resources
def download_nltk_resources():
    """Download required NLTK resources if not already available."""
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        print("Downloading NLTK vader_lexicon...")
        nltk.download('vader_lexicon', quiet=True)
        print("Download complete.")

# Download required resources
download_nltk_resources()

# Initialize sentiment analysis tools
vader_analyzer = SentimentIntensityAnalyzer()
gender_detector = gender.Detector()

def get_vader_sentiment(text):
    """
    Get sentiment scores for a given text using VADER sentiment analysis.

    Returns a list of four sentiment scores:
    - Negative sentiment score
    - Neutral sentiment score
    - Positive sentiment score
    - Compound sentiment score
    """
    sentiment_scores = vader_analyzer.polarity_scores(text)
    return [
        sentiment_scores['neg'],
        sentiment_scores['neu'],
        sentiment_scores['pos'],
        sentiment_scores['compound']
    ]

def get_gender(name):
    """
    Get gender prediction based on the name.

    Returns:
    - 0 for Male
    - 1 for Female
    """
    gender_result = gender_detector.get_gender(name)
    return 0 if gender_result == 'male' else 1

def encode_time(time):
    """Encode time by getting months away from ANCHOR_TIME
    """
    ANCHOR_TIME = datetime.strptime('2000-01-01', '%Y-%m-%d')
    if type(time) is str:
        time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')

    # get num months between ANCHOR_TIME and time
    return (ANCHOR_TIME.year - time.year) * 12 + ANCHOR_TIME.month - time.month

def save_dict_to_json(filename, data):
    """
    Save a dictionary to a JSON file.

    Parameters:
    - filename: Path to the output JSON file.
    - data: Dictionary to be saved.
    """
    with open(filename, "w") as outfile:
        json.dump(data, outfile, indent=4)

def load_dict_from_json(filename):
    """
    Load a dictionary from a JSON file.

    Parameters:
    - filename: Path to the input JSON file.

    Returns:
    - The dictionary loaded from the JSON file.
    """
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def hours_to_total_hours_per_week(hours):
    """
    Calculate the total hours a business is open per week based on its operating hours.

    Parameters:
    - hours (dict): A dictionary where keys are days of the week and values are strings representing the opening
                    and closing times in the format 'HH:MM-HH:MM'. If a day is closed or invalid, the value can be None or empty.

    Returns:
    - float: Total number of hours the business is open per week.
    """
    if not hours:
        return 0.0  # Return 0 if hours is empty or None

    total_hours_per_week = 0.0

    for day, timeframe in hours.items():
        if not timeframe:  # Skip days with no valid operating hours
            continue

        try:
            start_str, end_str = timeframe.split('-')
            start = datetime.strptime(start_str, '%H:%M')
            end = datetime.strptime(end_str, '%H:%M')

            # Handle cases where the time period crosses midnight
            if end < start:
                end += timedelta(days=1)  # Add a day to end time if it crosses midnight

            # Calculate time difference in hours
            time_diff = end - start
            total_hours_per_week += time_diff.total_seconds() / 3600  # Convert to hours
        except ValueError:
            # If the time format is incorrect, skip this entry and continue with the rest
            continue

    return total_hours_per_week

def encode_features_for_businesses(df, category_terms_list):
    """
    This function takes in a 'df' and encodes relevant features, returning the encoded dataframe, 'encoded_df'.
    """
    df = df[df['state'] == 'CA']  # Only consider states in California
    encoded_df = pd.DataFrame()
    # Add business_id
    encoded_df = df[['business_id']]

    # Identify & Add Numerical Features
    numerical_feats = ['stars', 'review_count', 'is_open', 'total_hours_per_week', 'num_checkins', 'compliment_count', 'latitude', 'longitude']
    encoded_df = pd.concat([encoded_df, df[numerical_feats]], axis=1)

    # Add Tip Sentiment
    encoded_df['mean_compound_sentiment'] = df['mean_compound_sentiment']
    encoded_df['mean_neg_sentiment'] = df['mean_neg_sentiment']
    encoded_df['mean_neu_sentiment'] = df['mean_neu_sentiment']
    encoded_df['mean_pos_sentiment'] = df['mean_pos_sentiment']

    # Add Categorical Features
    categorical_feats = ['address', 'city', 'postal_code', 'categories', 'hours']
    address_encoder = LabelEncoder()
    city_encoder = LabelEncoder()
    state_encoder = LabelEncoder()
    postal_code_encoder = LabelEncoder()
    encoded_df['address'] = address_encoder.fit_transform(df['address'])
    encoded_df['city'] = city_encoder.fit_transform(df['city'])
    encoded_df['postal_code'] = postal_code_encoder.fit_transform(df['postal_code'])
    encoded_df['state'] = state_encoder.fit_transform(df['state'])

    # Add Attributes
    attributes_lst = ['RestaurantsDelivery', 'OutdoorSeating', 'BusinessAcceptsCreditCards', 'BikeParking', 'RestaurantsPriceRange2', 'RestaurantsTakeOut', 'ByAppointmentOnly', 'WiFi', 'Alcohol', 'Caters']
    for attribute in attributes_lst:
        encoded_df[attribute] = df['attributes'].apply(
            lambda x: 1 if x and isinstance(x, dict) and attribute in x and x[attribute] == 'True' else 0
        )

    # Encode Categories
    for category in category_terms_list:
        encoded_df[category] = df['categories'].apply(lambda x: 1 if category in x else 0)

    return encoded_df

def clean_elite(s):
    """
    Given a string of comma separated dates (i.e. '2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,20,20,2021'),
    returns the number of individual dates.
    """
    if not s:
        return 0
    return s.count(",") + 1

def save_tensor(filename, data):
    """Function to save a tensor (or any other PyTorch object) to a file."""
    torch.save(data, filename)

def load_tensor(filename):
    """
    Load a PyTorch tensor from a file.
    
    Parameters:
    - filename: Path to the file
    
    Returns:
    - Loaded tensor
    """
    # Explicitly set weights_only=False as we are loading a graph object, not just weights
    return torch.load(filename, weights_only=False)