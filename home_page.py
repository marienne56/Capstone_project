import pandas as pd
import streamlit as st

from pages.display_user_list import display_users_list
from pages.update_account import fetch_user_details
# from update_account import fetch_user_details

# Initialisation de la session
if "view" not in st.session_state:
    st.session_state.view = "home"

if "identifier" not in st.session_state:
    st.session_state.identifier = None

if "client_name" not in st.session_state:
    st.session_state.client_name = None

if "role_name" not in st.session_state:
    st.session_state.role_name = None

# Sidebar - Navigation
identifier = st.session_state.identifier
user_details = fetch_user_details(identifier)
if user_details is not None and isinstance(user_details, pd.Series):
    client_name = user_details["ClientName"]
    st.sidebar.write(f"Welcome, :violet[{client_name}]!")
else:
    st.sidebar.error("Impossible de récupérer les informations de l'utilisateur.")

if st.sidebar.button("Home"):
    st.session_state.view = "home"


if st.sidebar.button("View Profil"):
   st.session_state.view = "display"
   display_users_list()



if st.sidebar.button("↩️Log Out"):
    st.session_state.view = "login" # Met à jour la vue actuelle
    st.switch_page("pages/login.py")  
    st.session_state.pop("identifier", None)  # Supprime les infos utilisateur
    st.session_state.pop("client_name", None)
    st.session_state.pop("role_name", None)
    st.rerun()  # Recharge l'interface pour rediriger vers la page de login


# Fonction d'affichage du dashboard (Accueil)
def home():
    st.write(f"Roles Name: {st.session_state.get('role_name', 'Unknown')}")
    
    # Vérifie si un utilisateur est connecté
    if "identifier" not in st.session_state or "client_name" not in st.session_state:
        st.sidebar.error("Utilisateur non connecté. Veuillez vous reconnecter.")
        st.write("Redirection en cours vers la page de connexion...")
        st.session_state.view = "login"
        st.rerun()
        return

    
    st.title("Dashboard")
    st.write("Ici, vous pouvez voir vos données et graphiques.")

# Fonction d'affichage de la gestion du compte
def account_management():
    st.title("Account Management")
    st.write("Ici, vous pouvez modifier vos informationsss.")
    st.session_state.page = "display"

# Ajouter un bouton engrenage en haut à droite via CSS
st.markdown("""
    <style>
        .gear-button {
            position: absolute;
            top: 10px;
            right: 10px;
            font-size: 2rem;
            cursor: pointer;
            background-color: transparent;
            border: none;
            color: #888;
        }
        .gear-button:hover {
            color: #555;
        }
    </style>
    <button class="gear-button" onclick="alert('You clicked the gear!')">&#9881;</button>
""", unsafe_allow_html=True)


if __name__ == "__main__":
     
# Affichage du contenu en fonction de `st.session_state.view`
    if st.session_state.view == "home":
        home()
    elif st.session_state.view == "account_management":
        account_management()
    #elif st.session_state.view == "login":
    # st.session_state.page = "display"
        #st.rerun

