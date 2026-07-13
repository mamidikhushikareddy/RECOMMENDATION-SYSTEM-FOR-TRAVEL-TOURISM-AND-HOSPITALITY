import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import folium
from folium.plugins import MarkerCluster
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# Load the datasets
hotel_data = pd.read_csv(r"C:\Users\mamid\Downloads\HOTELS LIST DATASET.csv")
user_suggestions = pd.read_csv(r"C:\Users\mamid\Downloads\HOTEL REVIEWS DATASET.csv")
travel_data = pd.read_csv(r"C:\Users\mamid\Downloads\goibibo trave sample.csv")

# Initialize global variables
user_search_history = {}

# Function to preprocess data
def preprocess_data(data):
    data['REVIEW'].fillna('', inplace=True)
    data['TOKENIZED_REVIEW'] = data['REVIEW'].apply(lambda x: word_tokenize(str(x).lower()))
    data['CLEANED_REVIEW'] = data['TOKENIZED_REVIEW'].apply(lambda x: ' '.join(x))
    return data

# Function to preprocess travel data
def preprocess_travel_data(data):
    # Add preprocessing steps for travel data if needed
    return data

# Function to recommend hotels based on user input price
def recommend_hotel(preferred_price):
    df = hotel_data.copy()
    X = df[['RATING', 'RATING COUNT', 'PRICE']].values
    knn_model = NearestNeighbors(n_neighbors=5, algorithm='auto')
    knn_model.fit(X)
    user_input = [[np.mean(df['RATING']), np.mean(df['RATING COUNT']), preferred_price]]
    _, indices = knn_model.kneighbors(user_input)
    recommended_hotels = df.iloc[indices[0]]
    return recommended_hotels

# Function to handle user input and display hotel details
def handle_search_button_click():
    preferred_price = st.sidebar.slider("Select your preferred price:", min_value=0, max_value=500, value=100, step=10)
    recommended_hotels = recommend_hotel(preferred_price)
    st.write("Recommended Hotels based on your preferred price:")
    st.write(recommended_hotels)

# Function to preprocess data for content-based filtering
def preprocess_data_content_based():
    global hotel_data
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(hotel_data['CLEANED_REVIEW'])
    return vectorizer, tfidf_matrix

# Function to recommend hotels based on user query (content-based filtering)
def recommend_hotels_content_based(user_query, vectorizer, tfidf_matrix):
    global hotel_data
    user_query_vector = vectorizer.transform([user_query])
    similarity_scores = cosine_similarity(user_query_vector, tfidf_matrix)
    similar_hotel_indices = similarity_scores.argsort()[0][-5:][::-1]
    unique_hotels = set()
    recommended_hotels = []
    for index in similar_hotel_indices:
        hotel_name = hotel_data.iloc[index]['HOTEL NAME']
        if hotel_name not in unique_hotels:
            unique_hotels.add(hotel_name)
            rating = hotel_data.iloc[index]['RATING']
            price = hotel_data.iloc[index]['PRICE']
            recommended_hotels.append((hotel_name, rating, price))
    return recommended_hotels

# Function to handle user input and recommend hotels (content-based filtering)
def handle_search_button_click_content_based():
    user_query = st.text_input("Enter your search query:")
    if user_query:
        vectorizer, tfidf_matrix = preprocess_data_content_based()
        recommended_hotels = recommend_hotels_content_based(user_query, vectorizer, tfidf_matrix)
        st.write("Recommended Hotels based on your search query:")
        st.write(recommended_hotels)

# Function to predict user rating based on collaborative filtering
def collaborative_filtering(user_id, user_query):
    global user_search_history
    similar_user_search_history = []
    for history_user_id, search_history in user_search_history.items():
        if history_user_id != user_id:
            similar_user_search_history.extend(search_history)
    if similar_user_search_history:
        filtered_data = hotel_data[hotel_data['CLEANED_REVIEW'].isin(similar_user_search_history)]
        average_rating = filtered_data['RATING'].mean()
    else:
        average_rating = hotel_data['RATING'].mean()
    return average_rating

# Function to display hotel reviews
def display_reviews(hotel_name, user_query):
    global hotel_data
    hotel_reviews = hotel_data[hotel_data['HOTEL NAME'] == hotel_name].copy()
    hotel_reviews['SENTIMENT'] = hotel_reviews['REVIEW'].apply(analyze_sentiment)
    reviews_html = generate_review_html(hotel_reviews)
    st.write(f"Reviews for {hotel_name} based on your search query '{user_query}':")
    st.write(reviews_html)

# Function to generate HTML for displaying reviews
def generate_review_html(hotel_reviews):
    reviews_html = ""
    for sentiment in ['positive', 'negative', 'neutral']:
        sentiment_reviews = hotel_reviews[hotel_reviews['SENTIMENT'] == sentiment]['REVIEW'].tolist()
        if sentiment_reviews:
            reviews_html += f"<h3>{sentiment.capitalize()} Reviews</h3>"
            reviews_html += "<ul>"
            for review in sentiment_reviews[:3]:  # Display only top 3 reviews
                reviews_html += f"<li>{review}</li>"
            reviews_html += "</ul>"
    return reviews_html

# Function to analyze sentiment of reviews
def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

# Function to visualize the distribution of recommended prices compared to user preferences
def plot_price_distribution(recommended_hotels, user_price):
    plt.figure(figsize=(8, 6))
    sns.histplot(recommended_hotels['PRICE'], color='skyblue', bins=20, kde=True, label='Recommended Prices')
    plt.axvline(x=user_price, color='red', linestyle='--', label='User Preferred Price')
    plt.title('Distribution of Recommended Prices vs. User Preferred Price')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Function to visualize the distribution of ratings for recommended hotels
def plot_rating_distribution(recommended_hotels):
    plt.figure(figsize=(8, 6))
    sns.histplot(recommended_hotels['RATING'], color='green', bins=10, kde=True)
    plt.title('Distribution of Ratings for Recommended Hotels')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.show()

# Function to visualize the distribution of rating counts for recommended hotels
def plot_rating_count_distribution(recommended_hotels):
    plt.figure(figsize=(8, 6))
    sns.histplot(recommended_hotels['RATING COUNT'], color='orange', bins=10, kde=True)
    plt.title('Distribution of Rating Counts for Recommended Hotels')
    plt.xlabel('Rating Count')
    plt.ylabel('Frequency')
    plt.show()

# Function to evaluate recommendations based on user preferences
def evaluate_recommendations(user_price):
    recommended_hotels = recommend_hotel(user_price)
    plot_price_distribution(recommended_hotels, user_price)
    plot_rating_distribution(recommended_hotels)
    plot_rating_count_distribution(recommended_hotels)

# Function to display user suggestions
def display_user_suggestions():
    st.write("User Suggestions:")
    st.write(user_suggestions)

# Function to get user location
def get_user_location():
    st.subheader("Enter Your Location")
    user_location = st.text_input("Enter your location (e.g., New York, USA):")
    if user_location:
        geolocator = Nominatim(user_agent="hotel_recommendation_system")
        location = geolocator.geocode(user_location)
        if location:
            return location.latitude, location.longitude
        else:
            st.error("Location not found. Please enter a valid location.")

# Function to display hotels on map
def display_hotels_on_map(user_location):
    st.subheader("Display Hotels on Map")
    if user_location:
        map_data = hotel_data[['HOTEL NAME', 'LATITUDE', 'LONGITUDE']]
        map_data['DISTANCE'] = map_data.apply(lambda x: geodesic((user_location[0], user_location[1]), (x['LATITUDE'], x['LONGITUDE'])).kilometers, axis=1)
        map_data = map_data.sort_values(by='DISTANCE')
        hotel_map = folium.Map(location=user_location, zoom_start=10)
        marker_cluster = MarkerCluster().add_to(hotel_map)
        for idx, row in map_data.iterrows():
            folium.Marker(
                location=[row['LATITUDE'], row['LONGITUDE']],
                popup=row['HOTEL NAME'],
                tooltip=f"Distance: {row['DISTANCE']:.2f} km",
            ).add_to(marker_cluster)
        st.write(hotel_map)

# Main function to run the Streamlit app
def main():
    st.title("Hotel Recommendation System")

    # Sidebar
    st.sidebar.title("Menu")
    menu_selection = st.sidebar.radio("Choose an option:", ["Recommendation", "User Suggestions"])

    if menu_selection == "Recommendation":
        st.subheader("Hotel Recommendation")

        # User input for hotel recommendation
        handle_search_button_click()
        handle_search_button_click_content_based()

        # Evaluation of recommendations
        st.subheader("Evaluation of Recommendations")
        evaluate_recommendations(100)  # Default price for evaluation

        # User location input for displaying hotels on map
        user_location = get_user_location()
        display_hotels_on_map(user_location)

    elif menu_selection == "User Suggestions":
        st.subheader("User Suggestions")
        display_user_suggestions()

    # Footer
    st.markdown("---")
    st.write("Built with Streamlit by [Your Name]")

if __name__ == "__main__":
    main()
