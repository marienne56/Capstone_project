import streamlit as st
import re
from dbConnection import get_connection
import hashlib
from sqlalchemy import text



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

def sign_up():

    if "page" not in st.session_state:
        st.session_state.page = "sign_upp"  # Page par défaut

    # Page par défaut
    st.title("Let's create your account")

    with st.form("registration_form", clear_on_submit=True):
        identifier = st.text_input("Identifier").strip()
        username = st.text_input("Username").strip()
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit_button = st.form_submit_button("Sign up")

    if submit_button:
        if not identifier or not username or not password:
            st.error("Please fill in all required fields")
            return

        if password != confirm_password:
            st.error("Passwords do not match!")
            return
        
        is_password_strong, password_message = validate_password_strength(password)
        if not is_password_strong:
            st.error(password_message)
            return

        try:
         with st.spinner('Registering... Please wait'):
            engine = get_connection()
            with engine.connect() as conn:
                # Vérifier si l'identifier existe dans la table consumption et récupérer ClientName
                check_query = text("SELECT ClientName, Ville, email, NumTel FROM consumption WHERE identifier = :identifier")


                result = conn.execute(check_query, {"identifier": identifier}).fetchone()

                if not result:
                    st.error("This identifier does not exist in the consumption table.")
                    return
                
                ClientName, Ville, email, NumTel = result if result else (None, None, None, None)


                # Vérifier si identifier ou username existe déjà dans users
                check_query = text("SELECT identifier, username FROM users WHERE identifier = :identifier OR username = :username")
                result = conn.execute(check_query, {"identifier": identifier, "username": username}).fetchone()

                if result:
                    if result.identifier == identifier:
                        st.error("This identifier is already registered.")
                    else:
                        st.error("This username is already taken.")
                    return

                # Insérer le nouvel utilisateur avec role par défaut "client"
                insert_query = text("""
                 INSERT INTO users (username, identifier, password_hash, role_id, ClientName, address, email, phone_number) 
                 VALUES (:username, :identifier, :password_hash, '456', :ClientName, :address, :email, :phone_number)
                """)

                conn.execute(insert_query, {
                    "username": username,
                    "identifier": identifier,
                    "password_hash": hash_password(password),
                    "ClientName": ClientName,
                    "address": Ville,
                    "email": email,
                    "phone_number": NumTel
                    
                })
                conn.commit()

                st.success("Registration successful! Redirecting to login page...")
                # st.session_state.page = "login"
                # st.rerun()

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please try again or contact support.")

    
 
    if st.button("🔙Log in"):
        st.session_state.page = "login"
                #st.switch_page("pages/login.py")
        st.rerun()


if __name__ == "__main__":
    sign_up()
