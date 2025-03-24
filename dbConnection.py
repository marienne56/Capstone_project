import streamlit as st
from sqlalchemy import create_engine

# Connexion à la base de données MySQL en utilisant SQLAlchemy
def get_connection():
    db_config = st.secrets["connections"]["mysql"]
    DB_CONNECTION = f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    engine = create_engine(DB_CONNECTION)
    return engine

