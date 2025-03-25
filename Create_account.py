import streamlit as st
import re
from dbConnection import get_connection
import hashlib
from sqlalchemy import text
#from utils import validate_email, validate_phone, validate_password_strength, hash_password, get_connection


def validate_email(email):
    """Validate email format using regex."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_phone(phone_number):
    """Validate phone number format."""
    pattern = r'^\+?1?\d{9,15}$'
    return re.match(pattern, phone_number) is not None

def hash_password(password):
    """Hash password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def validate_password_strength(password):
    """
    Validate password strength:
    - At least 8 characters
    - Contains uppercase and lowercase
    - Contains numbers
    - Contains special characters
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r"\d", password):
        return False, "Password must contain at least one number"
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain at least one special character"
    return True, "Password is strong"



# Add custom CSS
st.markdown("""
    <style>
        .stTextInput > label {font-weight: bold;}
        .stSelectbox > label {font-weight: bold;}
    </style>
""", unsafe_allow_html=True)


def sign_up_page():
 
    if "page" not in st.session_state:
        st.session_state.page = "sign_up"  # Page par défaut


 
    # Title of the application
    st.title("User Registration Form")

    # Ajout d'une clé pour le message de succès persistant
    if 'registration_success' not in st.session_state:
        st.session_state.registration_success = False

    # Pour conserver les données entre les soumissions du formulaire
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {
            'ClientName': '',
            'username': '',
            'email': '',
            'address': '',
            'is_active': 'Yes',
            'identifier': '',
            'phone_number': '',
            'role_name': 'Agent',
            'assigned_city': '',
        }

    # Afficher le message de succès s'il existe
    if st.session_state.registration_success:
        st.success("Registration successful!")
        # Reset du flag après affichage
        st.session_state.registration_success = False

    # Formulaire avec clear_on_submit=False
    with st.form("registration_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        
        with col1:
            ClientName = st.text_input("Name", value=st.session_state.form_data['ClientName']).strip()
            username = st.text_input("Username", value=st.session_state.form_data['username']).strip()
            email = st.text_input("Email", value=st.session_state.form_data['email']).strip()
            address = st.text_area("Address", value=st.session_state.form_data['address']).strip()
            is_active = st.selectbox("Is Active?", ["Yes", "No"], index=0 if st.session_state.form_data['is_active'] == "Yes" else 1)
            
        with col2:
            identifier = st.text_input("Identifier", value=st.session_state.form_data['identifier']).strip()
            phone_number = st.text_input("Phone Number", value=st.session_state.form_data['phone_number']).strip()
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
        
        # Définir les index pour le selectbox
        role_options = ["Agent", "Admin"]
        role_index = role_options.index(st.session_state.form_data['role_name']) if st.session_state.form_data['role_name'] in role_options else 0
        
        # Selectbox pour le rôle
        role_name = st.selectbox("Role Name", role_options, index=role_index)
        
        # Afficher le champ "Assigned City" UNIQUEMENT si le rôle est "Agent"
        if role_name == "Agent":
            assigned_city = st.text_input("Assigned City", value=st.session_state.form_data['assigned_city']).strip().lower()
        else:
            assigned_city = ""
        
        submit_button = st.form_submit_button("Create Account")

    if submit_button:
        # Mettre à jour les données de session
        st.session_state.form_data.update({
            'ClientName': ClientName,
            'username': username,
            'email': email,
            'address': address,
            'is_active': is_active,
            'identifier': identifier,
            'phone_number': phone_number,
            'role_name': role_name,
            'assigned_city': assigned_city
        })
        
        # Valider les champs requis
        required_fields = {
            'Name': ClientName,
            'Username': username,
            'Identifier': identifier,
            'Email': email,
            'Password': password,
        }
        
        # Ajouter Assigned City comme champ requis seulement si le rôle est Agent
        if role_name == "Agent":
            required_fields['Assigned City'] = assigned_city
        
        # Vérification des champs vides
        empty_fields = [field for field, value in required_fields.items() if not value]
        
        if empty_fields:
            st.error(f"Veuillez remplir les champs suivants: {', '.join(empty_fields)}")
        elif not validate_email(email):
            st.error("Please enter a valid email address")
        elif phone_number and not validate_phone(phone_number):
            st.error("Please enter a valid phone number")
        elif password != confirm_password:
            st.error("Passwords do not match!")
        else:
            # Validate password strength
            is_password_strong, password_message = validate_password_strength(password)
            if not is_password_strong:
                st.error(password_message)
            else:
                try:
                    engine = get_connection()
                    with engine.connect() as conn:
                        # Check if identifier, email, or username already exists
                        check_query = text("""
                            SELECT identifier, email, username 
                            FROM users 
                            WHERE identifier = :identifier OR email = :email OR username = :username
                        """)
                        
                        result = conn.execute(
                            check_query,
                            {
                                "identifier": identifier,
                                "email": email,
                                "username": username
                            }
                        ).fetchone()

                        if result:
                            # Provide specific feedback about which field exists
                            if result.identifier == identifier:
                                st.error("This identifier is already registered")
                            elif result.email == email:
                                st.error("This email is already registered")
                            else:
                                st.error("This username is already taken")
                        else:
                            # Get the role_id if role_name doesn't exist, insert it
                            role_id_query = text("SELECT role_id FROM role WHERE role_name = :role_name")
                            role_result = conn.execute(role_id_query, {"role_name": role_name}).fetchone()
                            if not role_result:
                                # Insert new role if not found
                                insert_role_query = text("""
                                    INSERT INTO role (role_name) 
                                    VALUES (:role_name)
                                """)
                                conn.execute(insert_role_query, {"role_name": role_name})
                                conn.commit()
                                role_id = conn.execute(role_id_query, {"role_name": role_name}).fetchone()[0]
                            else:
                                role_id = role_result[0]

                            # Insert new user
                            insert_query = text("""
                                INSERT INTO users (
                                    ClientName, username, identifier, 
                                    email, phone_number, address, password_hash, 
                                    is_active, role_id, assigned_city
                                ) VALUES (
                                    :ClientName, :username, :identifier,
                                    :email, :phone_number, :address, :password_hash,
                                    :is_active, :role_id, :assigned_city
                                )
                            """)
                            
                            conn.execute(
                                insert_query,
                                {
                                    "ClientName": ClientName,
                                    "username": username,
                                    "identifier": identifier,
                                    "email": email,
                                    "phone_number": phone_number,
                                    "address": address,
                                    "password_hash": hash_password(password),
                                    "is_active": is_active == "Yes",
                                    "role_id": role_id,
                                    "assigned_city": assigned_city
                                }
                            )
                            conn.commit()
                            

                            st.write("Avant réinitialisation:")
                            for key, value in st.session_state.form_data.items():
                                st.write(f"{key}: {value}")
                                                        # Vider le formulaire après l'enregistrement réussi
                           # Après l'enregistrement réussi, réinitialiser complètement les données du formulaire
                            st.session_state.form_data.clear()  # Effacer les données existantes
                            st.session_state.form_data.update({
                                'ClientName': '',
                                'username': '',
                                'email': '',
                                'address': '',
                                'is_active': 'Yes',  # Valeur par défaut
                                'identifier': '',
                                'phone_number': '',
                                'role_name': 'Agent',  # Rôle par défaut
                                'assigned_city': '',
                                "password_hash":'',
                            })

                            # Après réinitialisation
                            st.write("\nAprès réinitialisation:")
                            for key, value in st.session_state.form_data.items():
                                st.write(f"{key}: {value}")
                                            
                            # Définir le flag de succès à True pour afficher le message sur le prochain rendu
                            st.session_state.registration_success = True
                            
                            # Recharger la page
                            st.rerun()

                except Exception as e:
                    st.error(f"An error occurred during registration: {str(e)}")
                    st.error("Please try again or contact support if the problem persists.")

    # Bouton de retour
    if st.button("Back"):
        st.session_state.page = "display"
        st.rerun()

if __name__ == "__main__":
    sign_up_page()