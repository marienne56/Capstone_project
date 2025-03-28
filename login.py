import streamlit as st
import hashlib
from sqlalchemy import text
from dbConnection import get_connection
from  sign_up_client import sign_up
from dashboard import dashboard
from forgot_pw import password_change
from streamlit.runtime.scriptrunner import RerunException
from streamlit.runtime.scriptrunner import StopException
# Page par défaut

if "page" not in st.session_state:
        st.session_state.page = "login" 




# CSS pour améliorer le style
st.markdown("""
    <style>
        .stTextInput > label {font-weight: bold;}
        .stSelectbox > label {font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

def hash_password(password):
    """Hash password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def login_page():
    # Remove redundant session state initialization since it's already handled above
    if "page" not in st.session_state or st.session_state.page not in ["sign_up", "login", "home", "sign_upp", "display", "update", "forgot_password"]:
        st.session_state.page = "login"
    
    # Créer deux colonnes
    col1, col2 = st.columns([3, 1])  # Ajuste les proportions si nécessaire

    # Ajouter le titre dans la première colonne
    with col1:
        st.title("Hello! Let's :violet[Login]")

    # Ajouter l'image dans la deuxième colonne
    with col2:
        st.image("images/image.png", width=60)

    # Formulaire de connexion
    with st.form("login_form", clear_on_submit=True):
        identifier = st.text_input("Identifier (or Email or Username)").strip()
        password = st.text_input("Password", type="password")
       

        submit_button = st.form_submit_button("Loginn")

    if submit_button:
        if not identifier or not password:
            st.error("Please enter your identifier, email, or username and password.")
            return

        try:
            engine = get_connection()
            with engine.connect() as conn:
                # Requête pour vérifier si l'identifier, l'email ou le username existe avec le mot de passe
                check_query = text("""
                SELECT u.username, u.email, u.identifier, u.ClientName, r.role_name 
                FROM users u
                JOIN role r ON u.role_id = r.role_id
                WHERE (u.identifier = :identifier OR u.email = :identifier OR u.username = :identifier) 
                AND u.password_hash = :password_hash
                """)
                
                result = conn.execute(check_query, {
                    "identifier": identifier,
                    "password_hash": hash_password(password)  # Vérifier le hash du mot de passe
                }).fetchone()

                #st.write(f"DEBUG - Résultat SQL: {result}")  # Vérifier ce que renvoie la base de données

                if result:
                    st.session_state.client_name = result[3]  # Utilise l'index de 'ClientName'
                    
                    st.session_state.identifier = result[2]
                    st.session_state.role_name = result[4]
                    
                    
                    st.session_state.logged_in = True  # L'utilisateur est connecté
                    
                    

                    st.info("✅ You can now access your app! Enjoy your experience. 🚀")

                    
                    st.rerun()

                else:
                    st.error("This account does not exist or incorrect password.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please try again or contact support.")

    if not submit_button:
        

        if st.button("Forgot your password?"):
            #st.session_state.page = "forgot_password"
            
            st.session_state.page = "forgot_password"
            st.rerun()
        # Bouton pour aller vers l'inscription?

        col1, col2 = st.columns([1, 1])
        with col2:
            if st.button("Don't have an account? Please sign up"):
                st.session_state.page = "sign_upp"
                st.rerun()

if __name__ == "__main__":
    login_page()