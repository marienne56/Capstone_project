import streamlit as st
import pandas as pd
from account_management import fetch_users, fetch_user
from delete_user import delete_user
from dbConnection import get_connection
from pagination import fetch_total_users
from update_account import edit_user_page  # Import correct de la fonction d'édition
from sqlalchemy import text
from Create_account import sign_up_page 
from view_profil import view_profil


 # Page par défaut
if "page" not in st.session_state:
        st.session_state.page = "display" 

if "page_number" not in st.session_state:
        st.session_state.page_number = 1  

if "delete_confirmation" not in st.session_state:
             st.session_state["delete_confirmation"] = None


def get_role_name(role_id):
    """Récupère le nom du rôle à partir de l'ID du rôle."""
    try:
        engine = get_connection()
        with engine.connect() as conn:
            query = text("SELECT role_name FROM role WHERE role_id = :role_id")
            result = conn.execute(query, {"role_id": role_id}).fetchone()
            return result[0] if result else "Unknown"
    except Exception as e:
        st.error(f"Erreur lors de la récupération du rôle: {e}")
        return "Unknown"



# Stocker le rôle de l'utilisateur
if "role_id" in st.session_state:
    st.session_state.role_name = get_role_name(st.session_state.role_id)
    #st.write(f"Role Name: {st.session_state.role_name}")

# Initialize role_name if not set
if "role_name" not in st.session_state:
    st.session_state.role_name = "Unknown"



# Safely display role_name
#st.write(f"Role Name: {st.session_state.get('role_name', 'Unknown')}")
def display_users_list():
    if "identifier" not in st.session_state:
        st.session_state.identifier = None  # ou une valeur par défaut

    

    if "page_number" not in st.session_state:
        st.session_state.page_number = 1  # Valeur par défaut

    PAGE_SIZE = 2
    st.title("Account Management" if st.session_state.role_name == "Admin" else "My Profile")

    # Affichage du bouton Ajouter pour les admins
    if st.session_state.role_name == "Admin":
        #st.write("Admin has access")  # Vérifier si l'admin a bien accès
        col1, col2 = st.columns([8, 1])
        with col2:
            if st.button("➕"):
                st.session_state.page = "sign_up"
                st.rerun()

    # Barre de recherche pour les admins
    search_query = ""
    if st.session_state.role_name == "Admin":
        search_query = st.text_input("Search a user:", "").strip().lower()
       # st.write(f"Search query: {search_query}")  # Debug message
    # Gestion de l'édition d'un utilisateur
    query_params = st.query_params
    if "page" in query_params and query_params["page"] == "edit" and "identifier" in query_params:
        edit_user_page(query_params["identifier"])
        return

    # Récupération des utilisateurs
    if st.session_state.role_name == "Admin":
        df = fetch_users(search_query=search_query) if search_query else fetch_users(page_number=st.session_state.page_number)
    else:
        if "identifier" in st.session_state and st.session_state.identifier:
            df = fetch_user(st.session_state.identifier)
        else:
            st.error("No user identifier found. Please log in first.")
            st.stop()

    # Gestion des cas où il n'y a pas d'utilisateurs trouvés
    if df.empty:
             if search_query:
              st.warning(f"No user found for '{search_query}'.")
             elif "page_number" in st.session_state and st.session_state.page_number > 1:
              st.session_state.page_number -= 1
              st.rerun()
             else:
                st.warning("No user available.")

    # Affichage des utilisateurs
    headers = ["ID","Name", "Username", "Identifier", "Role", "Action"]
    #if st.session_state.role_name == "Admin":  # Afficher l'ID uniquement pour l'admin
        
    header_cols = st.columns([1, 2, 2, 2, 2, 3])
    for header_col, header in zip(header_cols, headers):
        header_col.write(f"**{header}**")

    df["ID"] = (st.session_state.page_number - 1) * PAGE_SIZE + df.index + 1 if not search_query else df.index + 1

    for _, row in df.iterrows():
        cols = st.columns([1, 2, 2, 2, 2, 3])
        cols[0].write(row["ID"])
        cols[1].write(row["ClientName"])
        cols[2].write(row["username"])
        cols[3].write(row["identifier"])
        cols[4].write(get_role_name(row["role_id"]))  # Afficher le role_name

        with cols[5]:
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("👁️", key=f"view_{row['identifier']}"):
                    
                   
                    st.write(f"Identifiant stocké: {st.session_state.get('edit_user_identifier', 'Aucun')}")
    
                    st.session_state["edit_user_identifier"] = row["identifier"]  # Assure-toi de sauvegarder l'identifiant
                    st.session_state.page = "view"
                    st.rerun() 



            with col2:
                if st.session_state.role_name == "Admin" and st.button("✏️", key=f"edit_{row['identifier']}"):
                    #edit_user_page(row["identifier"])
                    
                    #st.switch_page("update_account.py")  
                    st.session_state.edit_user_identifier = row["identifier"]
                    st.session_state.page = "update"
                    st.rerun()
                    st.write("hi")
            with col3:
                if st.session_state.role_name == "Admin" and st.button("🗑️", key=f"delete_{row['identifier']}"):
                    st.session_state["delete_confirmation"] = row["identifier"]
                    st.rerun()

    # Gestion de la confirmation de suppression
    if st.session_state.get("delete_confirmation"):
        identifier_to_delete = st.session_state["delete_confirmation"]
        st.warning(f"Do you really want to delete the user with identifier `{identifier_to_delete}`? This action is irreversible.")
        col_confirm, col_cancel = st.columns([1, 1])
        with col_confirm:
            if st.button("✅ Yes, delete"):
                try:
                    delete_user(identifier_to_delete)
                    st.session_state["delete_confirmation"] = None
                    st.success(f"The user `{identifier_to_delete}` has been deleted successfully.")
                except Exception as e:
                    st.error(f"Error deleting user: {e}")
                st.rerun()
        with col_cancel:
            if st.button("❌ Cancel"):
                st.session_state["delete_confirmation"] = None
                st.rerun()

    # Gestion de la pagination
    if st.session_state.role_name == "Admin":
        total_pages = 1
        page_number = st.session_state.get("page_number", 1)

        if not search_query:
            total_users = fetch_total_users()
            total_pages = (total_users // PAGE_SIZE) + (1 if total_users % PAGE_SIZE > 0 else 0)

        if total_pages > 1:
            col1, col2, col3 = st.columns([1, 6, 1])
            with col1:
                if page_number > 1 and st.button("⬅️"):
                    st.session_state.page_number -= 1
                    st.write(f"Going to page {st.session_state.page_number}")
                    st.rerun()
            with col2:
                page_selection = st.radio(" ", [str(i) for i in range(1, total_pages + 1)], horizontal=True, index=page_number - 1)
                new_page = int(page_selection)
                if new_page != page_number:
                    st.session_state.page_number = new_page
                    st.rerun()
            with col3:
                if page_number < total_pages and st.button("➡️"):
                  st.session_state.page_number += 1
                  st.rerun()
   


if __name__ == "__main__":
    display_users_list()