import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import pymysql  

# Load database credentials from secrets.toml
#db_config = st.secrets.connections.mysql
db_config = st.secrets["connections"]["mysql"]
# Accéder à la section des secrets pour MySQL

#db_config = st.secrets["connections.mysql"]

# Utilisation des informations pour se connecter à la base de données
DB_CONNECTION = f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"


# Create a database connection
engine = create_engine(DB_CONNECTION)

# Query database and load data

@st.cache_data
def load_data():
    query = "SELECT * FROM consumption LIMIT 10000;"  # Adjust query as needed
    return pd.read_sql(query, engine)

# Load and display data
st.title("MySQL Data Viewer")
df = load_data()
st.write(df)

