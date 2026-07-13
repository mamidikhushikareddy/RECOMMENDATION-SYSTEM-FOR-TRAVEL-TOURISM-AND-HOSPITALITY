import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Load the dataset
df = pd.read_csv(r"C:\Users\mamid\Downloads\HOTELS LIST DATASET.csv")

# Define features
features = ['RATING', 'RATING COUNT', 'PRICE']

# Extract feature values
X = df[features].values

# Fit the KNN model
knn_model = NearestNeighbors(n_neighbors=5, algorithm='auto')
knn_model.fit(X)

# Function to recommend hotels based on user input price
def recommend_hotel(preferred_price):
    # Transform user input
    user_input = [[np.mean(df['RATING']), np.mean(df['RATING COUNT']), preferred_price]]

    # Find nearest neighbors using cosine similarity
    _, indices = knn_model.kneighbors(user_input)

    # Get recommended hotels
    recommended_hotels = df.iloc[indices[0]]

    return recommended_hotels

def save_user_suggestion(user_id, user_name, hotel_name, suggested_price):
    # Implement the functionality to save user suggestions
    # For example, you can append the suggestion to a CSV file
    with open('user_suggestions.csv', 'a') as f:
        f.write(f"{user_id},{user_name},{hotel_name},{suggested_price}\n")


# Streamlit app
def main():
    st.title("Category-based Hotel Recommendation")
    
    # User input for preferred price
    preferred_price = st.sidebar.number_input('Preferred Price:', min_value=0, step=1, value=100)

    # Display recommended hotels based on preferred price
    recommended_hotels = recommend_hotel(preferred_price)
    st.write("Recommended Hotels:")
    st.write(recommended_hotels)

    # User suggestion form
    st.sidebar.title("User Price Suggestion")
    user_id = st.sidebar.text_input('User ID:')
    user_name = st.sidebar.text_input('User Name:')
    hotel_name = st.sidebar.selectbox('Select Hotel:', options=df['HOTEL NAME'].tolist())
    suggested_price = st.sidebar.number_input('Suggested Price:', min_value=0, step=1, value=100)

    # Save user suggestion to CSV file
    if st.sidebar.button('Add Suggestion'):
        save_user_suggestion(user_id, user_name, hotel_name, suggested_price)
        st.sidebar.success("Suggestion added successfully!")  # Display success message
    
    # Button to view suggested prices
    if st.button('View Suggested Prices'):
        try:
            user_suggestions = pd.read_csv('user_suggestions.csv')
            st.write("Suggested Prices:")
            st.write(user_suggestions)
        except FileNotFoundError:
            st.error("No suggested prices found!")  # Display error message if no suggestions are found

if __name__ == "__main__":
    main()

