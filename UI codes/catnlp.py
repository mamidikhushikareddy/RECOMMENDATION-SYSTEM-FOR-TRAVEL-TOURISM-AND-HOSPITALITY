import streamlit as st
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import pandas as pd

# Load Goibibo Indian hotel dataset
hotel_data_path = r"C:\Users\mamid\Downloads\goibibo trave sample.csv"

try:
    hotel_data = pd.read_csv(hotel_data_path)
except FileNotFoundError:
    st.error(f"File '{hotel_data_path}' not found.")
    st.stop()
except Exception as e:
    st.error(f"Error occurred while loading the CSV file: {e}")
    st.stop()

# Step 1: Data Preprocessing
def preprocess_data(hotel_data):
    try:
        # Convert text data to lowercase if columns exist
        if 'hotel_description' in hotel_data.columns:
            hotel_data['hotel_description'] = hotel_data['hotel_description'].str.lower()
        if 'hotel_facilities' in hotel_data.columns:
            hotel_data['hotel_facilities'] = hotel_data['hotel_facilities'].str.lower()
        if 'hotel_brand' in hotel_data.columns:
            hotel_data['hotel_brand'] = hotel_data['hotel_brand'].str.lower()
        # Combine text features for NLP
        if all(col in hotel_data.columns for col in ['hotel_brand', 'hotel_description', 'hotel_facilities']):
            hotel_data['text_features'] = hotel_data['hotel_brand'] + ' ' + hotel_data['hotel_description'] + ' ' + hotel_data['hotel_facilities']
    except Exception as e:
        st.error(f"Error occurred during data preprocessing: {e}")
        st.stop()
    return hotel_data


# Step 2: NLP-Based Keyword Search
def keyword_search(user_location, hotel_data, selected_categories):
    hotel_data['distance_to_user_km'] = hotel_data.apply(lambda row: geodesic(user_location, (row['latitude'], row['longitude'])).kilometers, axis=1)
    geolocator = Nominatim(user_agent="hotel_locator")
    user_location_info = geolocator.reverse(user_location)
    user_state = user_location_info.raw['address']['state']
    recommended_places = hotel_data[hotel_data['state'] == user_state]
    for category, value in selected_categories.items():
        if value == 'low':
            recommended_places = recommended_places[recommended_places[category] < recommended_places[category].quantile(0.33)]
        elif value == 'medium':
            recommended_places = recommended_places[(recommended_places[category] >= recommended_places[category].quantile(0.33)) & (recommended_places[category] < recommended_places[category].quantile(0.66))]
        elif value == 'high':
            recommended_places = recommended_places[recommended_places[category] >= recommended_places[category].quantile(0.66)]
        elif value == 'more':
            recommended_places = recommended_places[recommended_places[category] > recommended_places[category].mean()]
        elif value == 'less':
            recommended_places = recommended_places[recommended_places[category] < recommended_places[category].mean()]
    recommended_places = recommended_places.sort_values(by='distance_to_user_km')
    return recommended_places

def display_hotel_details(hotel, selected_categories):
    global hotel_counter
    if str(hotel['hotel_brand']) == 'nan':
        hotel_name = f"Hotel {hotel_counter}"
        hotel_counter += 1
    else:
        hotel_name = hotel['hotel_brand']
    
    details_dict = {
        "Hotel Name": [hotel_name]
    }
    
    for category in selected_categories:
        if category in hotel:
            details_dict[category.capitalize()] = [hotel[category]]
        else:
            details_dict[category.capitalize()] = ['N/A']
    
    details_df = pd.DataFrame(details_dict)
    st.dataframe(details_df)

# Initialize hotel counter
hotel_counter = 1

def main():
    # Preprocess the data
    hotel_data_processed = preprocess_data(hotel_data)

    # Create a title
    st.title("Category based Recommendation")

    # Arrange layout using columns
    col1, col2 = st.columns([6, 1])

    # Place inputs in the left column
    with col1:
        # Create text input widget for user location
        location_input = st.text_input("User Location:")

        # Define columns to exclude from the dropdown
        exclude_columns = ['crawl_date', 'image_count', 'page_url', 'qts', 'query_time_stamp', 'review_count_by_category']

        # Create multi-select dropdown widget for selecting categories with values
        categories_with_values = {col: ['low', 'medium', 'high', 'more', 'less'] for col in hotel_data_processed.columns if col not in exclude_columns}
        selected_categories = st.multiselect("Select Categories:", list(categories_with_values.keys()))

        # Create search button
        search_button = st.button("Search")

    # Perform search on button click
    if search_button:
        user_location_str = location_input

        # Geocode user location
        geolocator = Nominatim(user_agent="hotel_locator")
        user_location = geolocator.geocode(user_location_str)
        if user_location is None:
            st.error("Invalid user location. Please try again.")
            return
        user_coordinates = (user_location.latitude, user_location.longitude)

        # Get selected categories and their values
        selected_categories_values = {category: '' for category in selected_categories}

        # Perform keyword search
        recommendations = keyword_search(user_coordinates, hotel_data_processed, selected_categories_values)

        # Display hotel details for each recommendation
        for _, hotel in recommendations.iterrows():
            display_hotel_details(hotel, selected_categories)

if __name__ == "__main__":
    main()
