import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from faker_consumption_script import generate_synthetic_data
from datetime import datetime

def display_synthetic_data():
    st.title("Synthetic Data Viewer")
    
    # Create a sidebar with options
    st.sidebar.header("Generation Options")
    num_records = st.sidebar.slider("Number of records", 100, 5000, 1000, 100)
    
    start_year = st.sidebar.slider("Start year", 2020, 2025, 2023)
    end_year = st.sidebar.slider("End year", start_year, 2026, 2025)
    
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    
    cities = ["Yamoussoukro", "San Pedro", "Jacqueville", "Abidjan", "Man", 
              "Korhogo", "Séguela", "Katiola", "Boundiali", "Bouaké"]
    client_types = ["Residential", "Commercial", "Industrial"]
    
    # Generate data button
    if st.sidebar.button("Generate Data"):
        # Show loading spinner
        with st.spinner("Generating synthetic data..."):
            df = generate_synthetic_data(
                num_records=num_records,
                start_date=start_date,
                end_date=end_date,
                cities=cities,
                client_types=client_types
            )
            
            # Convert date columns
            df['periodeDebut'] = pd.to_datetime(df['periodeDebut'])
            df['periodeFin'] = pd.to_datetime(df['periodeFin'])
            
            # Add temporal analysis columns
            df['annee_debut'] = df['periodeDebut'].dt.year
            df['mois_debut'] = df['periodeDebut'].dt.month
            
            # Store the data in session state
            st.session_state.synthetic_data = df
            st.session_state.data_generated = True
    
    # If data has been generated
    if 'data_generated' in st.session_state and st.session_state.data_generated:
        df = st.session_state.synthetic_data
        
        # Data overview
        st.header("Data Overview")
        st.write(f"Generated {len(df)} records")
        
        # Display first 10 rows
        st.subheader("Sample Data (First 10 rows)")
        st.dataframe(df.head(10))
        
        # Basic statistics
        st.subheader("Basic Statistics")
        st.write("Consumption Statistics:")
        st.dataframe(df['Conso'].describe())
        
        # Distribution by client type
        st.subheader("Distribution by Client Type")
        client_counts = df['TypeClient'].value_counts()
        st.bar_chart(client_counts)
        
        # Distribution by city
        st.subheader("Distribution by City")
        city_counts = df['Ville'].value_counts()
        st.bar_chart(city_counts)
        
        # Distribution by payment status
        st.subheader("Distribution by Payment Status")
        payment_counts = df['StatutPaiement'].value_counts()
        st.bar_chart(payment_counts)
        
        # Time series visualization
        st.subheader("Consumption Over Time")
        
        # Aggregate by month and year
        time_data = df.groupby(['annee_debut', 'mois_debut'])['Conso'].mean().reset_index()
        time_data['date'] = pd.to_datetime(time_data['annee_debut'].astype(str) + '-' + time_data['mois_debut'].astype(str) + '-01')
        time_data = time_data.sort_values('date')
        
        # Plot time series
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_data['date'], time_data['Conso'], marker='o')
        ax.set_xlabel('Date')
        ax.set_ylabel('Average Consumption')
        ax.set_title('Average Consumption by Month')
        ax.grid(True)
        st.pyplot(fig)
        
        # Consumption by client type over time
        st.subheader("Consumption by Client Type Over Time")
        type_time_data = df.groupby(['annee_debut', 'mois_debut', 'TypeClient'])['Conso'].mean().reset_index()
        type_time_data['date'] = pd.to_datetime(type_time_data['annee_debut'].astype(str) + '-' + type_time_data['mois_debut'].astype(str) + '-01')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        for client_type in client_types:
            subset = type_time_data[type_time_data['TypeClient'] == client_type]
            ax.plot(subset['date'], subset['Conso'], marker='o', label=client_type)
        ax.set_xlabel('Date')
        ax.set_ylabel('Average Consumption')
        ax.set_title('Average Consumption by Client Type and Month')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        
        # Seasonal patterns
        st.subheader("Seasonal Patterns")
        season_data = df.groupby('mois_debut')['Conso'].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(season_data['mois_debut'], season_data['Conso'])
        ax.set_xlabel('Month')
        ax.set_ylabel('Average Consumption')
        ax.set_title('Average Consumption by Month (Seasonal Pattern)')
        ax.set_xticks(range(1, 13))
        ax.grid(True)
        st.pyplot(fig)
        
        # Download data option
        st.subheader("Download Data")
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="synthetic_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    display_synthetic_data()