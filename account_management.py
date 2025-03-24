import pandas as pd
from dbConnection import get_connection
from sqlalchemy import text
import streamlit as st

# Initialisation du numéro de page
if "page_number" not in st.session_state:
        st.session_state.page_number = 1 
    
    # Vérifier si la page est définie dans le session_state
if "page" not in st.session_state:
        st.session_state.page = "list"  # Page par défaut


if "page" not in st.session_state:
    st.session_state["page"] = "list"

if "delete_confirmation" not in st.session_state:
    st.session_state["delete_confirmation"] = None

if "page_number" not in st.session_state:
    st.session_state.page_number = 1

PAGE_SIZE = 2 
def fetch_user(identifier):
    try:
        engine = get_connection()
        with engine.connect() as conn:
            query = text('''
                SELECT users.identifier, users.ClientName, users.username, 
                       users.email, users.address, users.phone_number, users.is_active, 
                       users.role_id
                FROM users 
                WHERE users.identifier = :identifier
            ''')

            # Remplacement de ':identifier' par l'argument du paramètre
            df = pd.read_sql(query, conn, params={"identifier": identifier})

        if df.empty:
            st.error(f"The user with identifier {identifier} does not exist.")
        return df
    except Exception as e:
        st.error(f"Error retrieving user details : {str(e)}")
        return pd.DataFrame()
    
def fetch_users():
    try:
        engine = get_connection()
        with engine.connect() as conn:
            query = text('''
                SELECT 
    users.identifier, 
    users.ClientName,  
    users.username, 
    users.email, 
    users.address, 
    users.phone_number, 
    users.is_active, 
    users.role_id, 
    
    role.role_name
FROM 
    users
JOIN 
    role ON users.role_id = role.role_id
                        

            ''')

            df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Error retrieving user details : {str(e)}")
        return pd.DataFrame()
    


# Fonction pour récupérer les utilisateurs avec pagination
def fetch_users(page_number=None, search_query=None):
    try:
        engine = get_connection()
        with engine.connect() as conn:
            if search_query:
                search_query = search_query.strip().lower()
                query = text(f'''
                    SELECT
                        users.identifier,
                        users.ClientName,
                        
                        users.username,
                        users.email,
                        users.address,
                        users.phone_number,
                        users.is_active,
                        users.role_id,
                       
                        role.role_name
                    FROM
                        users
                    JOIN
                        role ON users.role_id = role.role_id
                    WHERE
                        LOWER(users.ClientName) LIKE :search_query OR
                       
                        LOWER(users.username) LIKE :search_query OR
                        LOWER(users.identifier) LIKE :search_query OR
                        LOWER(role.role_name) LIKE :search_query
                ''')
                df = pd.read_sql(query, conn, params={"search_query": f"%{search_query}%"})
            else:
                offset = (page_number - 1) * PAGE_SIZE if page_number else 0
                query = text(f'''
                    SELECT
                        users.identifier,
                        users.ClientName,
                        
                        users.username,
                        users.email,
                        users.address,
                        users.phone_number,
                        users.is_active,
                        users.role_id,
                        
                        role.role_name
                    FROM
                        users
                    JOIN
                        role ON users.role_id = role.role_id
                    LIMIT {PAGE_SIZE} OFFSET {offset}
                ''')
                df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Error retrieving user details : {str(e)}")
        return pd.DataFrame()
    
if __name__ == "__main__":
    fetch_users()