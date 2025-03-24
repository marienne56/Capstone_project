import streamlit as st
from dbConnection import get_connection
import pandas as pd
from sqlalchemy.sql import text

if "page" not in st.session_state:
        st.session_state.page = "view"



# Fonction pour récupérer les détails de l'utilisateur
def fetch_user_details(identifier):
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

            df = pd.read_sql(query, conn, params={"identifier": identifier})
            return df.iloc[0] if not df.empty else None
    except Exception as e:
        st.error(f"Error retrieving user details: {str(e)}")
        return None

# Fonction pour récupérer le nom du rôle
def fetch_role_name(role_id):
    try:
        engine = get_connection()
        with engine.connect() as conn:
            query = text('''SELECT role_name FROM role WHERE role_id = :role_id''')
            result = pd.read_sql(query, conn, params={"role_id": role_id})
            return result.iloc[0]['role_name'] if not result.empty else 'N/A'
    except Exception as e:
        st.error(f"Error retrieving role name: {str(e)}")
        return 'N/A'

# Fonction pour afficher le profil de l'utilisateur
def view_profil(identifier):
    user_details = fetch_user_details(identifier)  # Récupère les détails de l'utilisateur

    if user_details is not None:
        st.title(f"View the account's details of :violet[{user_details['ClientName']}] ")

        # Créer une disposition en deux colonnes pour une présentation claire
        col1, col2 = st.columns(2)

        # Colonne 1 : Informations générales
        with col1:
            st.subheader("Account's Informations")
            st.write(f"**Name:** {user_details.get('ClientName', 'N/A')}")
            st.write(f"**Username:** {user_details.get('username', 'N/A')}")
            st.write(f"**Email:** {user_details.get('email', 'N/A')}")
            st.write(f"**Identifier:** {user_details.get('identifier', 'N/A')}")

        # Colonne 2 : Informations supplémentaires
        with col2:
            st.subheader("Additional Informations")
            st.write(f"**Address:** {user_details.get('address', 'N/A')}")
            st.write(f"**Phone Number:** {user_details.get('phone_number', 'N/A')}")
            
            # Affichage de la valeur de "is_active"
            is_active = user_details.get('is_active', 0)
            st.write(f"**Is Active:** {'Yes' if is_active == 1 else 'No'}")
            
            # Récupérer et afficher le nom du rôle
            role_id = user_details.get('role_id', None)
            role_name = fetch_role_name(role_id)  # Appel de la fonction pour obtenir le rôle
            st.write(f"**Role:** {role_name}")
            # Add this after displaying the role name in the view_profil function
            if role_name == "Agent":
                try:
                    engine = get_connection()
                    with engine.connect() as conn:
                        assigned_city_query = text('''SELECT assigned_city FROM users WHERE identifier = :identifier''')
                        assigned_city_result = pd.read_sql(assigned_city_query, conn, params={"identifier": identifier})
                        assigned_city = assigned_city_result.iloc[0]['assigned_city'] if not assigned_city_result.empty else 'N/A'
                        st.write(f"**Assigned City:** {assigned_city}")
                except Exception as e:
                    st.error(f"Error retrieving assigned city: {str(e)}")
    else:
        st.error(f"The user with the identifier {identifier} does not exist.")



    # Ajoute un bouton pour revenir à la liste des utilisateurs
    if st.button("🔙Back"):
        # Mettre à jour l'état de la session pour revenir à la page "list"
        st.session_state.page = "display"  # "display" est la page qui affiche la liste des utilisateurs
        #st.switch_page("display_user_list.py")
        st.session_state.page_number = 1  # Optionnel : réinitialiser la page à la première page si nécessaire
        st.rerun()  # Recharger la page pour refléter la mise à jour de l'état


if __name__ == "__main__":
    identifier = st.session_state.get("identifier", None)
    if identifier:
        view_profil(identifier)
    else:
        st.error("No identifier found. Please log in first.")