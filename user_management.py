import streamlit as st
import pandas as pd
from dbConnection import get_connection
from pages.update_account import edit_user_page  # Import correct de la fonction d'édition
from sqlalchemy import text
from Create_account import sign_up_page 

st.title("Account Management")

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

# Fonction pour récupérer les utilisateurs par identifier
def fetch_user(identifier):
    try:
        engine = get_connection()
        with engine.connect() as conn:
            query = text('''
                SELECT users.identifier, users.first_name, users.last_name, users.username, 
                       users.email, users.address, users.phone_number, users.is_active, 
                       users.role_id, users.zone_id 
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
    users.first_name, 
    users.last_name, 
    users.username, 
    users.email, 
    users.address, 
    users.phone_number, 
    users.is_active, 
    users.role_id, 
    users.zone_id, 
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
                        users.first_name,
                        users.last_name,
                        users.username,
                        users.email,
                        users.address,
                        users.phone_number,
                        users.is_active,
                        users.role_id,
                        users.zone_id,
                        role.role_name
                    FROM
                        users
                    JOIN
                        role ON users.role_id = role.role_id
                    WHERE
                        LOWER(users.first_name) LIKE :search_query OR
                        LOWER(users.last_name) LIKE :search_query OR
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
                        users.first_name,
                        users.last_name,
                        users.username,
                        users.email,
                        users.address,
                        users.phone_number,
                        users.is_active,
                        users.role_id,
                        users.zone_id,
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


# Fonction pour récupérer le nombre total d'utilisateurs
def fetch_total_users():
    try:
        engine = get_connection()
        with engine.connect() as conn:
            query = text('''SELECT COUNT(*) FROM users''')
            result = conn.execute(query).fetchone()
            total_users = result[0]
        return total_users
    except Exception as e:
        st.error(f"Error retrieving user details : {str(e)}")
        return 0



def delete_user(identifier):
    try:
        engine = get_connection()
        with engine.connect() as conn:
            query = text('DELETE FROM users WHERE identifier = :identifier')
            conn.execute(query, {"identifier": identifier})
            conn.commit()  # Appliquer les changements
        st.success(f"The user {identifier} has been deleted successfully.")
    except Exception as e:
        st.error(f"Erreur lors de la suppression de l'utilisateur : {str(e)}")

# Fonction principale pour afficher la liste des utilisateurs


# Initialiser les variables de session
if "delete_confirmation" not in st.session_state:
    st.session_state["delete_confirmation"] = None
if "page_number" not in st.session_state:
    st.session_state.page_number = 1  # Commence à la page 1

PAGE_SIZE = 2  # Nombre d'utilisateurs par page

def main():

    
    # Si le bouton "Add" est cliqué, on redirige vers la page de création de compte
    if st.session_state.page != "sign_up":  # Afficher "Add" uniquement si ce n'est pas la page d'inscription
     if st.session_state.page == "list":
       col1, col2 = st.columns([8,1])

       with col2:
        if st.button("➕"):
         st.session_state.page = "sign_up"
         st.rerun()  # Relancer l'application pour appliquer le changement# Relancer l'application pour appliquer le changement

    # Afficher la page en fonction de la valeur de st.session_state.page
    if st.session_state.page == "sign_up":
        sign_up_page()
    
    else:
        # Barre de recherche
        search_query = st.text_input("Search an user :", "").strip().lower()

        # Vérifie si l'URL contient les paramètres nécessaires pour l'édition
        query_params = st.query_params
        if "page" in query_params and query_params["page"] == "edit" and "identifier" in query_params:
            identifier = query_params["identifier"]
            edit_user_page(identifier)  # Appeler la fonction d'édition depuis update.py
        else:
            # Récupération des utilisateurs pour la page actuelle avec filtre de recherche
            if search_query:
                df = fetch_users(search_query=search_query)
            else:
                page_number = st.session_state.page_number
                df = fetch_users(page_number=page_number)

            # Vérifier si la page est vide et revenir en arrière si nécessaire
            if df.empty:
             if search_query:
              st.warning(f"No users found for '{search_query}'.")
             elif "page_number" in st.session_state and st.session_state.page_number > 1:
              st.session_state.page_number -= 1
              st.rerun()
             else:
                st.warning("No user available.")
 

            # Affichage des utilisateurs
            if not df.empty:
                headers = ["ID", "First Name", "Last Name", "Username", "Identifier", "Role", "Actions"]
                header_cols = st.columns([1, 2, 2, 2, 2, 2, 3])
                for header_col, header in zip(header_cols, headers):
                    header_col.write(f"**{header}**")

                df['ID'] = (st.session_state.page_number - 1) * PAGE_SIZE + df.index + 1 if not search_query else df.index + 1
                for index, row in df.iterrows():
                    with st.container():
                        cols = st.columns([1, 2, 2, 2, 2, 2, 3])
                        cols[0].write(row["ID"])
                        cols[1].write(row["first_name"])
                        cols[2].write(row["last_name"])
                        cols[3].write(row["username"])
                        cols[4].write(row["identifier"])
                        cols[5].write(row["role_name"])

                        with cols[6]:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.button("👁️", key=f"view_{row['identifier']}")
                            with col2:
                                if st.button("✏️", key=f"edit_{row['identifier']}"):
                                    st.query_params["page"] = "edit"
                                    st.query_params["identifier"] = row['identifier']
                                    st.rerun()
                            with col3:
                                if st.button("🗑️", key=f"delete_{row['identifier']}"):
                                    st.session_state["delete_confirmation"] = row['identifier']
                                    st.rerun()

            # Affichage de la confirmation de suppression
            if st.session_state["delete_confirmation"]:
                identifier_to_delete = st.session_state["delete_confirmation"]
                st.warning(f"Do you really want the user with identifier `{identifier_to_delete}` ? This action is irreversible")

                col_confirm, col_cancel = st.columns([1, 1])
                with col_confirm:
                    if st.button("✅ Yes, delete"):
                        delete_user(identifier_to_delete)
                        st.session_state["delete_confirmation"] = None
                        st.success(f"the user with identifier `{identifier_to_delete}` has been successfully deleted.")
                        st.rerun()
                with col_cancel:
                    if st.button("❌ Cancel"):
                        st.session_state["delete_confirmation"] = None
                        st.rerun()

            # Gestion de la pagination
            if not search_query:
             total_users = fetch_total_users()
             total_pages = (total_users // PAGE_SIZE) + (1 if total_users % PAGE_SIZE > 0 else 0)

            if total_pages > 1:
             col1, col2, col3 = st.columns([1, 6, 1])

            with col1:
               if page_number > 1:
                if st.button("⬅️"):
                    st.session_state.page_number -= 1
                    st.rerun()

            with col2:
    # Affichage des numéros de page
             pages_display = [str(i) for i in range(1, total_pages + 1)]
             page_selection = st.radio(" ", pages_display, horizontal=True, index=page_number - 1)

    # Mise à jour de la page si l'utilisateur clique sur un numéro
             new_page = int(page_selection)
             if new_page != page_number:
              st.session_state.page_number = new_page
              st.rerun()

        with col3:
            if page_number < total_pages:
                if st.button("➡️"):
                    st.session_state.page_number += 1
                    st.rerun()


if __name__ == "__main__":
    main()




                

