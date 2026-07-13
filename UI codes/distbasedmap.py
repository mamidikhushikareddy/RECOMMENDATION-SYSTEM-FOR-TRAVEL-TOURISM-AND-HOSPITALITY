import streamlit as st
import numpy as np
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from sklearn.feature_extraction.text import TfidfVectorizer

# Load Goibibo Indian hotel dataset
fdata = pd.read_csv(r"C:\Users\mamid\Downloads\goibibo trave sample.csv")

# Data cleaning & feature extraction
# You can put your data cleaning code here...

# Step 1: Data Preprocessing
def preprocess_data(hotel_data):
    # Handle missing values
    hotel_data.dropna(inplace=True)
    # Clean text data
    hotel_data['hotel_description'].fillna('', inplace=True)
    hotel_data['hotel_facilities'].fillna('', inplace=True)
    # Combine text features for NLP
    hotel_data['text_features'] = hotel_data['hotel_brand'] + ' ' + hotel_data['room_type'] + ' ' + hotel_data['hotel_description'] + ' ' + hotel_data['hotel_facilities']
    return hotel_data

# Step 2: NLP-Based Keyword Search
def keyword_search(user_location, tfidf_matrix, tfidf_vectorizer, hotel_data, selected_categories=None):
    # Calculate distances from user location to hotel locations
    hotel_data['distance_to_user_km'] = hotel_data.apply(lambda row: geodesic(user_location, (row['latitude'], row['longitude'])).kilometers, axis=1)

    # Get user state
    geolocator = Nominatim(user_agent="hotel_locator")
    user_location_info = geolocator.reverse(user_location)
    user_state = user_location_info.raw['address']['state']

    # Filter places based on state
    recommended_places = hotel_data[hotel_data['state'] == user_state]

    # Sort places based on distance from user location
    recommended_places = recommended_places.sort_values(by='distance_to_user_km')

    return recommended_places

# Step 3: Map Visualization with Color-Coded Ratings
def display_recommendations_on_map(recommendations, user_location):
    # Create a map centered around the search location
    map_places = folium.Map(location=user_location, zoom_start=12)

    # Add avatar marker for user location with a colorful icon
    folium.Marker(location=user_location, popup="Your Location", icon=folium.Icon(color='green', icon='user')).add_to(map_places)

    # Create MarkerClusters for different rating groups
    ratings = sorted(recommendations['hotel_star_rating'].unique(), reverse=True)
    for rating in ratings:
        rating_group = recommendations[recommendations['hotel_star_rating'] == rating]
        marker_cluster = MarkerCluster().add_to(map_places)

        # Add markers for each place in the rating group
        for index, row in rating_group.iterrows():
            place_location = (row['latitude'], row['longitude'])
            popup_html = f"<b>{row['hotel_brand']}</b><br><i>Address:</i> {row['address']}<br><i>Distance to Your Location:</i> {row['distance_to_user_km']:.2f} km"
            popup_html += f"<br><i>Rating:</i> {row['hotel_star_rating']}"
            folium.Marker(location=place_location, popup=popup_html).add_to(marker_cluster)

    # Display the map
    st.write(map_places._repr_html_(), unsafe_allow_html=True)

def main():
    # Preprocess the data
    hotel_data_processed = preprocess_data(fdata)

    # TF-IDF vectorization for text features
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(hotel_data_processed['text_features'])

    # Create text input widget for user location
    location_input = st.text_input("Enter your location (e.g., city, state, country):")

    # Create search button
    search_button = st.button("Search")

    # Define function to handle search button click
    if search_button:
        # Geocode user location
        geolocator = Nominatim(user_agent="hotel_locator")
        user_location = geolocator.geocode(location_input)
        if user_location is None:
            st.error("Invalid user location. Please try again.")
            return
        user_coordinates = (user_location.latitude, user_location.longitude)

        # Perform keyword search without selected_categories
        recommendations = keyword_search(user_coordinates, tfidf_matrix, tfidf_vectorizer, hotel_data_processed)

        # Display recommendations on map
        display_recommendations_on_map(recommendations, user_coordinates)

        # Display additional information below the map
        st.subheader("Recommended Places:")
        st.write(recommendations[['hotel_brand', 'address', 'hotel_star_rating', 'distance_to_user_km']])

if __name__ == "__main__":
    main()