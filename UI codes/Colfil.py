import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime

# Load the dataset
data = pd.read_csv(r"C:\Users\mamid\Downloads\HOTEL REVIEWS DATASET.csv")

# Preprocess the dataset (cleaning, handling missing values, etc.)
data['REVIEW'].fillna('', inplace=True)

# Tokenize user reviews
data['TOKENIZED_REVIEW'] = data['REVIEW'].apply(lambda x: word_tokenize(str(x).lower()))

# Combine tokenized reviews into strings
data['CLEANED_REVIEW'] = data['TOKENIZED_REVIEW'].apply(lambda x: ' '.join(x))

# Vectorize the reviews using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['CLEANED_REVIEW'])

# Dictionary to store user search history and their corresponding ratings
user_search_history = {}

# Function to interpret user search query and recommend hotels
def recommend_hotels(user_id, user_query, data, vectorizer, tfidf_matrix):
    # Store user search history
    if user_id not in user_search_history:
        user_search_history[user_id] = {'queries': [], 'ratings': [], 'timestamps': []}

    # Store the current search query and timestamp
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_search_history[user_id]['queries'].append(user_query)
    user_search_history[user_id]['timestamps'].append(current_time)

    # Vectorize the user query using TF-IDF
    user_query_vector = vectorizer.transform([user_query])

    # Calculate similarity between user query and hotels
    similarity_scores = cosine_similarity(user_query_vector, tfidf_matrix)

    # Get the indices of hotels with similar keywords in reviews as the search query
    similar_hotel_indices = similarity_scores.argsort()[0][-5:][::-1]

    # Retrieve the hotel details for unique hotels
    unique_hotels = set()
    recommended_hotels = []
    for index in similar_hotel_indices:
        hotel_name = data.iloc[index]['1 NAME']
        if hotel_name not in unique_hotels:
            unique_hotels.add(hotel_name)
            rating = data.iloc[index]['RATING']
            price = data.iloc[index]['PRICE']
            similarity_score = similarity_scores[0][index]
            recommended_hotels.append((hotel_name, rating, price, similarity_score))

    return recommended_hotels

# Function to predict user search history rating
def predict_rating(user_id):
    # Generate a random rating for each user
    return np.random.rand()

# Function to perform sentiment analysis on a text
def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

# Function to display reviews for a particular hotel
def display_reviews(hotel_name, user_query, data):
    hotel_reviews = data[data['1 NAME'] == hotel_name]
    hotel_reviews['SENTIMENT'] = hotel_reviews['REVIEW'].apply(analyze_sentiment)
    reviews_html = ""
    for sentiment in ['positive', 'negative', 'neutral']:
        sentiment_reviews = hotel_reviews[hotel_reviews['SENTIMENT'] == sentiment]['REVIEW'].tolist()
        if sentiment_reviews:
            reviews_html += f"### {sentiment.capitalize()} Reviews\n"
            for review in sentiment_reviews[:3]:  # Display only top 3 reviews
                reviews_html += f"* {review}\n"
    st.write(f"## Reviews for {hotel_name}")
    st.write(f"**Search Query:** {user_query}")
    st.markdown(reviews_html)

# Main Streamlit app
def main():
    user_id = st.sidebar.text_input("Enter user ID:")
    user_query = st.sidebar.text_input("Enter your search query:")

    if st.sidebar.button("Search"):
        # Recommend hotels based on the user query
        recommended_hotels = recommend_hotels(user_id, user_query, data, vectorizer, tfidf_matrix)
        st.write(f"Recommended Hotels based on search query: {user_query}")
        for i, (hotel_name, rating, price, similarity_score) in enumerate(recommended_hotels):
            st.write(f"## {hotel_name}")
            st.write(f"**Rating:** {rating}")
            st.write(f"**Price:** {price}")
            st.write(f"**Similarity Score:** {similarity_score}")

            # Display reviews for the hotel
            display_reviews(hotel_name, user_query, data)

        # Predict user search history rating
        mse = predict_rating(user_id)
        st.sidebar.markdown(f"**Predicted User Search History Rating:** <span style='color: blue;'>{mse}</span>", unsafe_allow_html=True)

        # Update user search history with ratings
        user_search_history[user_id]['ratings'].append(mse)  # Update user rating with predicted MSE

        # Update the user search history file after every search
        user_search_history_df = pd.DataFrame.from_dict(user_search_history, orient='index')
        user_search_history_df.to_csv('user_search_history.csv', mode='a', header=False)  # Append new entries to the file

if __name__ == "__main__":
    main()
