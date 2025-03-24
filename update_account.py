import streamlit as st
from dbConnection import get_connection
import pandas as pd
from sqlalchemy.sql import text

if "page" not in st.session_state:
        st.session_state.page = "update"

# Fonction pour récupérer les détails de l'utilisateur à partir de la base de données
def fetch_user_details(identifier):
    try:
        engine = get_connection()
        with engine.connect() as conn:
            query = text('''
    SELECT users.identifier, users.ClientName, users.username, 
           users.email, users.address, users.phone_number, users.is_active, 
           users.role_id, users.assigned_city
    FROM users 
    WHERE users.identifier = :identifier
''')

            df = pd.read_sql(query, conn, params={"identifier": identifier})
            return df.iloc[0] if not df.empty else None
    except Exception as e:
        st.error(f"Error retrieving user details : {str(e)}")
        return None


# Add this function to your file with edit_user_page
def fetch_role_name(role_id):
    try:
        engine = get_connection()
        with engine.connect() as conn:
            query = text('''SELECT role_name FROM role WHERE role_id = :role_id''')
            result = conn.execute(query, {"role_id": role_id}).fetchone()
            #st.write(f"Debug - SQL query result: {result}")  # Débogage
            return result[0] if result else 'N/A'
    except Exception as e:
        st.error(f"Error retrieving role name: {str(e)}")
        return 'N/A'
# Fonction pour afficher la page de modification d'un utilisateur
def edit_user_page(identifier):
    user_details = fetch_user_details(identifier)  # Récupère les détails de l'utilisateur

    if user_details is not None:
        st.title(f"Edit the account's details of  {user_details['ClientName']} ")
    # Créer un formulaire pour modifier les informations de l'utilisateur
        with st.form(key='edit_user_form'):
            col1, col2 = st.columns(2)
            with col1:
                ClientName = st.text_input("Name", value=user_details.get('ClientName', ''))
                username = st.text_input("Username", value=user_details.get('username', ''))
                email = st.text_input("Email", value=user_details.get('email', ''))
                address = st.text_area("Address", value=user_details.get('address', ''))

            with col2:
                identifier = st.text_input("Identifier", value=user_details.get('identifier', ''))
                phone_number = st.text_input("Phone Number", value=user_details.get('phone_number', ''))
                is_active = st.selectbox("Is Active?", ["Yes", "No"], index=0 if user_details.get('is_active', False) else 1)
            
            # Get current role name
            current_role_name = fetch_role_name(user_details.get('role_id'))

            # Create role selection with current role pre-selected
            role_options = ["Client", "Agent", "Admin"]
            role_index = role_options.index(current_role_name) if current_role_name in role_options else 0
            role_name = st.selectbox("Role ID", role_options, index=role_index)

        # Show assigned_city only for Agents
            if role_name == "Agent":
                assigned_city = st.text_input("Assigned City", value=user_details.get('assigned_city', ''))
            else:
                assigned_city = ""
   

            # 🛠️ IMPORTANT : Le bouton doit être aligné correctement !
            submit_button = st.form_submit_button("Update")

        # Traitement après soumission du formulaire
        if submit_button:
                update_user_details(
                    identifier, ClientName, username, email, address,
                    phone_number, is_active == "Yes", role_name, assigned_city if role_name == "Agent" else ""
                )

    else:
        st.error(f"The user with the identifier {identifier}does not exist.")
        
    # Ajoute un bouton pour revenir à la liste des utilisateurs
    if st.button("Back to the users list"):
        # Mettre à jour l'état de la session pour revenir à la page "list"
        st.session_state.page = "display"  # "display" est la page qui affiche la liste des utilisateurs
        #st.switch_page("display_user_list.py")

        st.session_state.page_number = 1  # Optionnel : réinitialiser la page à la première page si nécessaire
        st.rerun()  # Recharger la page pour refléter la mise à jour de l'état

# Fonction pour mettre à jour les informations de l'utilisateur
def update_user_details(identifier, ClientName, username, email, address, phone_number, is_active, role_name, assigned_city = ""):
    try:
        engine = get_connection()
        with engine.connect() as conn:
            # Insérer le role_name dans la table role si ce n'est pas déjà présent
            role_id_query = text("SELECT role_id FROM role WHERE role_name = :role_name")
            role_result = conn.execute(role_id_query, {"role_name": role_name}).fetchone()
            if not role_result:
                # Insérer un nouveau role si il n'existe pas
                insert_role_query = text("INSERT INTO role (role_name) VALUES (:role_name)")
                conn.execute(insert_role_query, {"role_name": role_name})
                conn.commit()
                role_id = conn.execute(role_id_query, {"role_name": role_name}).fetchone()[0]
            else:
                role_id = role_result[0]

            # Insérer le zone_name dans la table zone si ce n'est pas déjà présent
            '''
            zone_id_query = text("SELECT zone_id FROM zone WHERE zone_name = :zone_name")
            zone_result = conn.execute(zone_id_query, {"zone_name": zone_name}).fetchone()
            if not zone_result:
                # Insérer une nouvelle zone si elle n'existe pas
                insert_zone_query = text("INSERT INTO zone (zone_name) VALUES (:zone_name)")
                conn.execute(insert_zone_query, {"zone_name": zone_name})
                conn.commit()
                zone_id = conn.execute(zone_id_query, {"zone_name": zone_name}).fetchone()[0]
            else:
                zone_id = zone_result[0]
            '''
            # Requête SQL de mise à jour des données
            update_query = text(''' 
                UPDATE users
                SET ClientName = :ClientName, username = :username, email = :email,
                    address = :address, phone_number = :phone_number, is_active = :is_active, 
                    role_id = :role_id, assigned_city = :assigned_city
                WHERE identifier = :identifier
            ''')

            #st.write("Paramètres envoyés à la mise à jour :", params)

            # Si un mot de passe est fourni, on l'ajoute à la requête
            params = {
                "ClientName": ClientName,  "username": username, "email": email,
                "address": address, "phone_number": phone_number, "is_active": is_active,
                "role_id": role_id, "assigned_city": assigned_city, "identifier": identifier
            }
            
            # Exécution de la requête de mise à jour
            conn.execute(update_query, params)
            conn.commit()

            st.success("User details successfully updated.")
    except Exception as e:
        st.error(f"Update error :  : {str(e)}")
if __name__ == "__main__":
    identifier = st.session_state.get("identifier", None)
    if identifier:
        edit_user_page(identifier)
    else:
        st.error("No identifier found. Please log in first.")