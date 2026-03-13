import streamlit as st
import pandas as pd
import folium
import re
from streamlit_folium import folium_static
import json

# Constants
DEFAULT_CATEGORIES = ['Mean Income']

def clean_data(data):
    """Clean and format the column names in the data."""
    data.columns = [re.sub(r'(_)', ' ', col).title() for col in data.columns]
    return data

def load_data():
    """Load the dataset and geojson information."""
    data = pd.read_csv('GCSE results by sex - 2023-24.csv')
    with open('london_boroughs.geojson', 'r') as f:
        geo_data = json.load(f)
    return data, geo_data

def identify_borough_column(new_data, existing_boroughs):
    """Identify which column in the new data likely contains borough names."""
    matching_counts = {}
    for col in new_data.columns:
        if new_data[col].dtype == 'object':
            matching_counts[col] = sum(new_data[col].isin(existing_boroughs))
    likely_borough_column = max(matching_counts, key=matching_counts.get)
    return likely_borough_column

def identify_and_merge_borough_column(existing_data, new_data):
    """Identify the borough column in the new data and merge the datasets."""
    borough_col = identify_borough_column(new_data, existing_data['Borough'])
    new_data.rename(columns={borough_col: "Borough"}, inplace=True)
    new_data = new_data[new_data['Borough'].isin(existing_data['Borough'])]
    columns_to_append = st.sidebar.multiselect(
        'Select columns to append from the new file',
        new_data.columns.difference(['Borough']),
        default=[]
    )
    merged_data = existing_data.merge(
    new_data[['Borough'] + columns_to_append], on='Borough', how='left')
    return merged_data

def load_new_data(existing_data):
    """Load new data from uploaded file and append selected columns to existing data."""
    uploaded_file = st.sidebar.file_uploader("Upload a new file (CSV or Excel):",
    type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            new_data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            new_data = pd.read_excel(uploaded_file)
        else:
            st.sidebar.write("Unsupported file type. Please upload a CSV or Excel file.")
            return existing_data
        existing_data = identify_and_merge_borough_column(existing_data, new_data)
    return existing_data

def display_selected_data(data, selected_boroughs, selected_categories):
    """Display the data and bar chart visualizations for selected boroughs and categories."""
    selected_data = data[data['Borough'].isin(selected_boroughs)][['Borough'] + selected_categories]
    st.write(selected_data)
    for category in selected_categories:
        st.subheader(category)
        st.bar_chart(selected_data[['Borough', category]].set_index('Borough'))
        display_summary_statistics(selected_data, category)

def display_summary_statistics(data, category):
    """Display summary statistics for a given category."""
    max_val = data[category].max()
    min_val = data[category].min()
    mean_val = data[category].mean()
    median_val = data[category].median()
    borough_max = data.at[data[category].idxmax(), 'Borough']
    borough_min = data.at[data[category].idxmin(), 'Borough']
    st.write(f"Max {category}: {max_val} (Borough: {borough_max})")
    st.write(f"Min {category}: {min_val} (Borough: {borough_min})")
    st.write(f"Mean {category}: {mean_val:.2f}")
    st.write(f"Median {category}: {median_val}")

def create_map(data, selected_boroughs, selected_categories, geo_data):
    """Create and display a folium map."""
    m = folium.Map(location=[51.5074, -0.1278], zoom_start=10)
    folium.GeoJson(geo_data, name="geojson").add_to(m)
    for borough in selected_boroughs:
        row = data[data['Borough'] == borough].iloc[0]
        tooltip_lines = [f"{category}: {row[category]}" for category in selected_categories]
        tooltip_text = f"{row['Borough']}\\n" + "\\n".join(tooltip_lines)
        folium.Marker(
            [row['Latitude'], row['Longitude']],
            tooltip=tooltip_text,
            icon=folium.Icon(color="red")
        ).add_to(m)
    folium_static(m)

def display_sidebar(data):
    """Display sidebar selectors for user input and return user selections."""
    user_input_borough = st.sidebar.text_input("Type the borough of your choice:", value="")
    default_boroughs = [user_input_borough] if user_input_borough in data['Borough'].unique() else []
    selected_boroughs = st.sidebar.multiselect('Select Boroughs to Compare',
    data['Borough'].unique(), default=default_boroughs)
    selected_categories = st.sidebar.multiselect('Select Categories',
    data.columns.difference(['Borough', 'Latitude', 'Longitude']),
    default=DEFAULT_CATEGORIES)
    return selected_boroughs, selected_categories

def main():
    """Main function to run the Streamlit app."""
    st.title("London Boroughs Analysis")
    data, geo_data = load_data()
    data = clean_data(data) # Clean the data after loading
    data = load_new_data(data)
    selected_boroughs, selected_categories = display_sidebar(data)
    if not selected_boroughs or not selected_categories:
        st.info("Please select at least one borough and one criteria from the sidebar.")
        return
    create_map(data, selected_boroughs, selected_categories, geo_data)
    display_selected_data(data, selected_boroughs, selected_categories)

if __name__ == '__main__':
    main()