import streamlit as st
import pandas as pd
from login import login_page
from streamlit_option_menu import option_menu
from dashboard import dashboard
from update_account import edit_user_page
from view_profil import view_profil
from forgot_pw import password_change 
from sign_up_client import sign_up
from Create_account import sign_up_page
from display_user_list import display_users_list
from update_account import edit_user_page, fetch_user_details
from view_profil import view_profil
from prediction_models import prediction_dashboard, load_prediction_data, load_synthetic_data
from permission_system import check_admin_permission
from dbConnection import get_connection
from sqlalchemy.sql import text
from update_account import fetch_role_name

# Initialisation des variables de session
if "role_name" not in st.session_state:
    st.session_state.role_name = None

if "identifier" not in st.session_state:
    st.session_state.identifier = None

if "page" not in st.session_state:
    st.session_state.page = "login"

# Style CSS conservé de votre code original
st.markdown("""
    <style>
        /* Masquer le bouton Deploy */
        .st-emotion-cache-1wbqy5l,
        [data-testid="stDeployButton"],
        button[kind="primary"],
        .stDeployButton,
        iframe[title="Deploy"],
        div[data-testid="stToolbar"] button {
            display: none !important;
            width: 0 !important;
            height: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            border: none !important;
            pointer-events: none !important;
            position: absolute !important;
            top: -9999px !important;
            left: -9999px !important;
            opacity: 0 !important;
            visibility: hidden !important;
            clip: rect(0 0 0 0) !important;
            overflow: hidden !important;
        }

        /* Cibler le bouton de menu avec les trois points et garder visible */
        div[data-testid="stToolbar"] button[aria-label="Options"] {
            visibility: visible !important;
            opacity: 1 !important;
            position: relative !important;
            top: auto !important;
            left: auto !important;
            pointer-events: all !important;
        }

        /* S'assurer que le bouton de menu reste interactif */
        div[data-testid="stToolbar"] button[aria-label="Options"]:hover {
            background-color: #f4f4f4;
        }
    </style>
    
    <script>
        function forceDisableButton() {
            const selectors = [
                '.st-emotion-cache-1wbqy5l',
                '[data-testid="stDeployButton"]',
                'button[kind="primary"]',
                '.stDeployButton',
                'iframe[title="Deploy"]',
                'div[data-testid="stToolbar"] button'
            ];
            
            selectors.forEach(selector => {
                const elements = document.querySelectorAll(selector);
                elements.forEach(element => {
                    if (element) {
                        element.remove();
                    }
                });
            });

            document.addEventListener('click', function(e) {
                const target = e.target;
                if (target && 
                    (target.matches(selectors.join(',')) || 
                     target.closest(selectors.join(',')))) {
                    e.preventDefault();
                    e.stopPropagation();
                    return false;
                }
            }, true);
        }

        forceDisableButton();
        
        const observer = new MutationObserver(forceDisableButton);
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
        
        setInterval(forceDisableButton, 100);
    </script>
""", unsafe_allow_html=True)

def main():
    # Contrôle de l'affichage de la sidebar en fonction de l'état de connexion
    if st.session_state.identifier is None:
        # Utilisateur non connecté, afficher uniquement le formulaire de login
        if st.session_state.page == "login":
            login_page()
        elif st.session_state.page == "sign_upp":
            sign_up()
        elif st.session_state.page == "sign_up":
            sign_up_page()
        elif st.session_state.page == "forgot_password":
            password_change()
        return  # Ne pas continuer avec le reste de la fonction
    
    # À partir d'ici, l'utilisateur est connecté, afficher la sidebar
    with st.sidebar:

        #st.logo( "images/gs2eci_logo.jpg", size="large", link=None, icon_image=None)
        st.image("images/gs2eci_logo.jpg", width=60)

        # Récupérer les informations de l'utilisateur pour l'affichage
        if st.session_state.client_name:
        #     st.sidebar.write(f"Welcome, :violet[{st.session_state.client_name}]!")
         # Vérifier si l'utilisateur est admin
            is_admin = check_admin_permission(st.session_state.identifier)
                
                # Afficher l'en-tête
            if is_admin:
                    st.sidebar.header(f"Welcome, Admin :violet[{st.session_state.client_name}]!")
            else:
                    st.sidebar.header(f"Welcome, :violet[{st.session_state.client_name}]!")
               
        # Menu de navigation
        selected = option_menu(
            menu_title=None,
            options=[" 📈 Analysis", " 🔮 Prediction", "👨🏾 View accounts", " ↩️ Log out"],
            menu_icon="cast",
            default_index=0,  # Analysis comme page par défaut
        )
    
    # Traitement des options de menu pour les utilisateurs connectés
    if selected == " 📈 Analysis":
        dashboard()
    
    elif selected == " 🔮 Prediction":
        user_details = fetch_user_details(st.session_state.identifier)
        if user_details is not None and isinstance(user_details, pd.Series):
            client_name = user_details["ClientName"]
            
            
            # Titre de la page
            st.title("🔮 Prediction of data consumption of a SODECI client")
            
            # Chargement des données (toujours nécessaire)
            df = load_prediction_data()  # Par défaut, charger les données réelles
            
            # L'option des données synthétiques uniquement pour admin
            use_synthetic_data = False
            if is_admin:
                # Option pour utiliser des données synthétiques (admin seulement)
                st.sidebar.write("🔍 Filters for prediction")
                use_synthetic_data = st.sidebar.checkbox("Use synthetic data for predictions", value=False)
                
                if use_synthetic_data:
                    # Si l'admin choisit d'utiliser des données synthétiques
                    n_synthetic_records = st.sidebar.slider(
                        "Number of synthetic records", 
                        min_value=1000, 
                        max_value=10000, 
                        value=5000, 
                        step=500
                    )
                    st.sidebar.info(f"Using {n_synthetic_records} synthetic records")
                    
                    # Charger les données synthétiques
                    df = load_synthetic_data(n_synthetic_records)
            
            # Vérifiez si les données ont été chargées
            if df.empty:
                st.error("⚠️ Unable to load data. Please check your database connection.")
                return
            
            # Obtenir directement le rôle de l'utilisateur depuis la base de données
            user_role = "N/A"  # Valeur par défaut
            try:
                engine = get_connection()
                with engine.connect() as conn:
                    query = text("""
                        SELECT r.role_name 
                        FROM users u 
                        JOIN role r ON u.role_id = r.role_id 
                        WHERE u.identifier = :identifier
                    """)
                    result = conn.execute(query, {"identifier": st.session_state.identifier}).fetchone()
                    if result and result[0]:
                        user_role = result[0]
            except Exception as e:
                st.error(f"Error checking user role: {str(e)}")
            
            # Filtres communs pour tous les utilisateurs
            # Options d'année - pour tous les utilisateurs
            available_years = sorted(df['annee_debut'].unique(), reverse=True)
            selected_year = st.sidebar.selectbox("📅 Year", available_years)
            
            # Options de prédiction - pour tous les utilisateurs
            prediction_timeframe = st.sidebar.radio(
                "Predict for:",
                ["The next semester", "The next quarter", "The next month"]
            )
            
            # Filtres uniquement pour admin
            if is_admin:
                # Filtre utilisateur - uniquement pour les admins
                all_users = ["All users"] + sorted(df['identifier'].unique())
                selected_user = st.sidebar.selectbox("👤 Users", all_users)
            else:
                # Pour les utilisateurs non-admin, utiliser leur propre identifiant
                selected_user = st.session_state.identifier
            
            # Appliquer les filtres
            df_filtered = df.copy()
            assigned_city = None
            
            # Appliquer le filtre utilisateur
            if selected_user != "All users" and is_admin:
                df_filtered = df_filtered[df_filtered['identifier'] == selected_user]
            elif not is_admin and user_role != "Agent":
                # Utilisateurs non-admin non-agents ne voient que leurs données
                df_filtered = df_filtered[df_filtered['identifier'] == selected_user]
            
            # Traitement spécial pour les agents
            if user_role == "Agent":
                #st.write("Agent role detected")
                # Query the database to get the assigned city for this agent
                try:
                    engine = get_connection()
                    with engine.connect() as conn:
                        query = text('''SELECT assigned_city FROM users WHERE identifier = :identifier''')
                        result = conn.execute(query, {"identifier": st.session_state.get("identifier", "")}).fetchone()
                        if result and result[0]:
                            assigned_city = result[0]
                            
                            # Afficher la ville assignée pour débogage
                            #st.write(f"Agent assigned to city: '{assigned_city}'")
                            
                            # Vérifier si la colonne 'Ville' existe
                            if 'Ville' not in df_filtered.columns:
                                st.error("Column 'Ville' not found in data. Available columns: " + ", ".join(df_filtered.columns))
                            else:
                                # Afficher les villes disponibles avant filtrage
                                available_cities = df_filtered['Ville'].unique()
                                #st.write(f"Available cities in data: {available_cities}")
                                
                                # Approche 1: Correspondance exacte (insensible à la casse)
                                mask_exact = df_filtered['Ville'].str.lower() == assigned_city.lower()
                                
                                # Approche 2: Correspondance partielle
                                mask_partial = df_filtered['Ville'].str.lower().str.contains(assigned_city.lower())
                                
                                # Approche 3: Correspondance avec la première partie du nom
                                mask_startswith = df_filtered['Ville'].str.lower().str.startswith(assigned_city.lower())
                                
                                
                                # Utiliser la meilleure approche disponible
                                if sum(mask_exact) > 0:
                                    df_filtered = df_filtered[mask_exact]
                                    #st.success(f"Using exact matches for '{assigned_city}'")
                                elif sum(mask_partial) > 0:
                                    df_filtered = df_filtered[mask_partial]
                                    #st.success(f"Using partial matches for '{assigned_city}'")
                                elif sum(mask_startswith) > 0:
                                    df_filtered = df_filtered[mask_startswith]
                                    #st.success(f"Using 'starts with' matches for '{assigned_city}'")
                                else:
                                    # Si aucune correspondance n'est trouvée, afficher un avertissement
                                    st.warning(f"No matching cities found for '{assigned_city}'. Using all available data.")
                                
                                # NOUVEAU CODE: Sélecteur de client pour les agents
                                if not df_filtered.empty:
                                    # Récupérer la liste des identifiants de clients de cette ville
                                    client_identifiers = sorted(df_filtered['identifier'].unique())
                                    #st.write(f"Found {len(client_identifiers)} clients in {assigned_city}")
                                    
                                    # Ajouter un sélecteur dans la sidebar pour choisir un client spécifique
                                    st.sidebar.write("### Clients in your assigned city")
                                    view_all_clients = st.sidebar.checkbox("View aggregated data for all clients", value=True)
                                    
                                    if not view_all_clients:
                                        selected_client = st.sidebar.selectbox(
                                            "Select a specific client:",
                                            ["All clients"] + client_identifiers
                                        )
                                        
                                        # Si un client spécifique est sélectionné, filtrer davantage les données
                                        if selected_client != "All clients":
                                            df_filtered = df_filtered[df_filtered['identifier'] == selected_client]
                                            st.success(f"Showing predictions for client: {selected_client}")
                                    #else:
                                        #st.success(f"Showing aggregated predictions for all clients in {assigned_city}")
                except Exception as e:
                    st.error(f"Error retrieving or processing assigned city: {str(e)}")
                    st.exception(e)  # Affiche la trace complète de l'erreur
            
            # Filtre par année (pour tous)
            if selected_year is not None:
                df_filtered = df_filtered[df_filtered['annee_debut'] == selected_year]
            
            # Vérifier si nous avons des données après filtrage
            if df_filtered.empty:
                st.warning("⚠️ No data matches selected filters.")
                return
            
            # Convertir le type de prédiction au format attendu
            period_type = ""
            if "month" in prediction_timeframe.lower():
                period_type = "Monthly"
            elif "quarter" in prediction_timeframe.lower():
                period_type = "Quarterly"
            elif "semester" in prediction_timeframe.lower():
                period_type = "Semi-annual"
            else:
                period_type = "Annual"
            
            # Appeler le dashboard de prédiction avec les permissions
            prediction_dashboard(
                df_filtered, 
                selected_user,
                selected_year,
                prediction_timeframe,
                is_synthetic=use_synthetic_data,
                is_admin=is_admin,
                user_role=user_role,
                assigned_city=assigned_city
            )
    
    elif selected == "👨🏾 View accounts":
        display_users_list()
    
    elif selected == " ↩️ Log out":
        # Supprimer l'identifier de la session
        if "identifier" in st.session_state:
            del st.session_state["identifier"]
        
        # Réinitialiser les autres variables de session
        st.session_state.page = "login"
        if "client_name" in st.session_state:
            del st.session_state["client_name"]
        if "role_name" in st.session_state:
            del st.session_state["role_name"]
        if "logged_in" in st.session_state:
            del st.session_state["logged_in"]
        
        # Afficher un message de confirmation
        st.success("You have been logged out.")
        
        # Forcer le rechargement pour revenir à la page de login sans sidebar
        st.rerun()
    
    # Gestion des autres états de page (view, update, etc.)
    if st.session_state.page == "view":
        identifier = st.session_state.get("edit_user_identifier", None)
        if identifier:
            view_profil(identifier)
        else:
            st.error("No user selected for modification.")
    
    elif st.session_state.page == "update":
        identifier = st.session_state.get("edit_user_identifier", None)
        if identifier:
            edit_user_page(identifier)
        else:
            st.error("No user selected for modification.")
    
    elif st.session_state.page == "sign_upp":
        sign_up()
    
    elif st.session_state.page == "sign_up":
        sign_up_page()
    
    elif st.session_state.page == "forgot_password":
        password_change()

if __name__ == "__main__":
    main()