import streamlit as st
from dbConnection import get_connection
from sqlalchemy import text


def fetch_total_users():
    try:
        engine = get_connection()
        with engine.connect() as conn:
            query = text('''SELECT COUNT(*) FROM users''')
            result = conn.execute(query).fetchone()
            total_users = result[0]
        return total_users
    except Exception as e:
        st.error(f"Error retrieving user totals : {str(e)}")
        return 0
    

if __name__ == "__main__":
    fetch_total_users()  