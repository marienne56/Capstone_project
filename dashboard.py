import streamlit as st
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime, timedelta
import calendar
from dbConnection import get_connection  # Import the connection function
from display_user_list import display_users_list
from update_account import fetch_user_details
from permission_system import check_admin_permission, get_user_role
# st.set_page_config(
#     page_title="SODECI Consumption Analysis",
#     page_icon="💧",
#     layout="wide"
# )
from sqlalchemy.sql import text

if "identifier" not in st.session_state:
        st.session_state.identifier = None  # ou une valeur par défaut
def dashboard():
    identifier = st.session_state.identifier

    if not identifier:
        st.error("Please Login first")
        return

    # Vérifier le rôle de l'utilisateur
    is_admin = check_admin_permission(identifier)

        # Ajouter ce code:
    # Vérifier si l'utilisateur est un agent et obtenir sa ville assignée
    user_role = "N/A"
    assigned_city = None
    try:
        engine = get_connection()
        with engine.connect() as conn:
            # Récupérer le rôle de l'utilisateur
            query = text("""
                SELECT r.role_name 
                FROM users u 
                JOIN role r ON u.role_id = r.role_id 
                WHERE u.identifier = :identifier
            """)
            result = conn.execute(query, {"identifier": identifier}).fetchone()
            if result and result[0]:
                user_role = result[0]
                
                # Si c'est un agent, récupérer sa ville assignée
                if user_role == "Agent":
                    query = text('''SELECT assigned_city FROM users WHERE identifier = :identifier''')
                    result = conn.execute(query, {"identifier": identifier}).fetchone()
                    if result and result[0]:
                        assigned_city = result[0]
    except Exception as e:
        st.error(f"Error checking user role: {str(e)}")

    # App title
    st.title("📊 SODECI Customer Consumption Analysis")

    # Affichage du nom de l'utilisateur
    user_details = fetch_user_details(identifier)
    if user_details is not None and isinstance(user_details, pd.Series):
        client_name = user_details["ClientName"]
        # if is_admin:
        #     st.sidebar.header(f"Welcome, Admin :violet[{client_name}]!")
        # else:
        #     st.sidebar.header(f"Welcome, :violet[{client_name}]!")

        # Utility functions
        def get_quarter(date):
            """Returns the quarter (1-4) for a given date"""
            return (pd.to_datetime(date).month - 1) // 3 + 1

        def get_semester(date):
            """Returns the semester (1-2) for a given date"""
            return 1 if pd.to_datetime(date).month <= 6 else 2

        def get_quarter_label(quarter):
            """Converts a quarter number to label"""
            return f"Q{quarter}"

        def get_semester_label(semester):
            """Converts a semester number to label"""
            return f"S{semester}"

        def get_week_of_month(date):
            """Calculates the week of the month (1-5) for a given date"""
            date = pd.to_datetime(date)
            first_day = date.replace(day=1)
            dom = date.day
            adjusted_dom = dom + first_day.weekday()
            return (adjusted_dom - 1) // 7 + 1

        def quarter_to_months(quarter):
            """Returns the months corresponding to a quarter"""
            if quarter == 1:
                return ["Jan", "Feb", "Mar"]
            elif quarter == 2:
                return ["Apr", "May", "Jun"]
            elif quarter == 3:
                return ["Jul", "Aug", "Sep"]
            else:
                return ["Oct", "Nov", "Dec"]
            
        def generate_interpretation(df, period_type, grouping_column=None, value_column='Conso', group_label=None):
            """Generate interpretation text for a graph based on the data"""
            if df.empty:
                return "Insufficient data available for interpretation."
                
            # Get total value
            total = df[value_column].sum()
            
            # Find max and min periods
            max_row = df.loc[df[value_column].idxmax()]
            min_row = df.loc[df[value_column].idxmin()]
            
            if grouping_column:
                max_period = max_row[group_label] if group_label in max_row else max_row[grouping_column]
                min_period = min_row[group_label] if group_label in min_row else min_row[grouping_column]
            else:
                max_period = "the selected period"
                min_period = "the selected period"
            
            max_value = max_row[value_column]
            min_value = min_row[value_column]
            
            # Calculate average
            avg_value = df[value_column].mean()
            
            # Calculate percentage difference between max and min
            if min_value > 0:
                percent_diff = ((max_value - min_value) / min_value) * 100
            else:
                percent_diff = 100  # To avoid division by zero
                
            # Generate basic interpretation
            if value_column == 'Conso':
                metric = "consumption"
                unit = "units"
            else:  # 'MontFact'
                metric = "billing"
                unit = "FCFA"
            
            interpretation = f"**📊 Interpretation:** The total {metric} for this period is {total:,.1f} {unit}. "
            interpretation += f"The highest {metric} occurred in {max_period} ({max_value:,.1f} {unit}), "
            interpretation += f"while the lowest was in {min_period} ({min_value:,.1f} {unit}). "
            
            if percent_diff > 50:
                interpretation += f"There is a significant variation of {percent_diff:.1f}% between the highest and lowest values, "
                interpretation += "indicating substantial fluctuations in the data. "
            elif percent_diff > 20:
                interpretation += f"There is a moderate variation of {percent_diff:.1f}% between the highest and lowest values. "
            else:
                interpretation += f"The data shows relatively stable {metric} with only {percent_diff:.1f}% variation between highest and lowest values. "
                
            # Add trend analysis if applicable
            if len(df) > 2 and grouping_column:
                # Simple trend analysis
                first_value = df.iloc[0][value_column]
                last_value = df.iloc[-1][value_column]
                
                if last_value > first_value:
                    percent_increase = ((last_value - first_value) / first_value) * 100 if first_value > 0 else 100
                    interpretation += f"There is an overall increasing trend of approximately {percent_increase:.1f}% over the period."
                elif last_value < first_value:
                    percent_decrease = ((first_value - last_value) / first_value) * 100 if first_value > 0 else 100
                    interpretation += f"There is an overall decreasing trend of approximately {percent_decrease:.1f}% over the period."
                else:
                    interpretation += "The overall trend remains stable over the period."
                    
            return interpretation

        # Data loading
        @st.cache_data   
        def load_data():
            try:
                engine = get_connection()
                
                # Execute SQL query to retrieve all data from the consumption table
                query = "SELECT * FROM consumption"
                df = pd.read_sql(query, engine)
                
                # Date conversions
                df['periodeDebut'] = pd.to_datetime(df['periodeDebut'])
                df['periodeFin'] = pd.to_datetime(df['periodeFin'])
                
                # Add temporal analysis columns
                df['annee_debut'] = df['periodeDebut'].dt.year
                df['trimestre_debut'] = df['periodeDebut'].apply(get_quarter)
                df['semestre_debut'] = df['periodeDebut'].apply(get_semester)
                df['mois_debut'] = df['periodeDebut'].dt.month
                df['nom_mois_debut'] = df['periodeDebut'].dt.month_name()
                
                # Add week of month
                df['semaine_mois'] = df['periodeDebut'].apply(get_week_of_month)
                
                # For more precise analysis, we can also consider the end period
                df['annee_fin'] = df['periodeFin'].dt.year
                df['trimestre_fin'] = df['periodeFin'].apply(get_quarter)
                df['semestre_fin'] = df['periodeFin'].apply(get_semester)
                
                # Calculate billing duration in days
                df['duree_facturation'] = (df['periodeFin'] - df['periodeDebut']).dt.days
                
                # Calculate average consumption per day
                df['conso_par_jour'] = df['Conso'] / df['duree_facturation'].replace(0, 1)  # Avoid division by zero
                
                return df
            except Exception as e:
                st.error(f"Error loading data: {e}")
                return pd.DataFrame()
    
        # Chargement des données - UNE SEULE FOIS
        with st.spinner('Loading data... ⏳'):
            all_data = load_data()
        
        # Vérifier si les données ont été chargées
        if all_data.empty:
            st.error("⚠️ Unable to load data. Please check database connection.")
            return
            
        if is_admin:
            df = all_data  # Admin voit toutes les données
        elif user_role == "Agent" and assigned_city:
            # Les agents voient les données des clients de leur ville assignée
            import unicodedata
            
            # Fonction pour normaliser les textes
            def normalize_text(text):
                if not isinstance(text, str):
                    return ""
                # Convertir en minuscules
                text = text.lower()
                # Supprimer les espaces en début et fin
                text = text.strip()
                # Supprimer les accents
                text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
                return text
            
            # Normaliser la ville assignée
            normalized_assigned_city = normalize_text(assigned_city)
            
            # Créer une colonne temporaire avec les villes normalisées
            all_data['normalized_ville'] = all_data['Ville'].apply(normalize_text)
            
            # Filtrer les données
            df = all_data[all_data['normalized_ville'] == normalized_assigned_city]
            
            # Supprimer la colonne temporaire
            df = df.drop('normalized_ville', axis=1)
            
            if df.empty:
                st.warning(f"⚠️ No data found for your assigned city: {assigned_city}")
                return
            
            st.info(f"Showing data for clients in your assigned city: {assigned_city}")
        else:
            # Utilisateurs normaux ne voient que leurs données
            df = all_data[all_data['identifier'] == identifier]
            if df.empty:
                st.warning(f"⚠️ No data found for your account (identifier: {identifier}).")
                return
        
        # Sidebar pour les filtres
        st.sidebar.write("🔍 Filters")
        # Filtre par utilisateur - pour admin ET agents
        if is_admin or user_role == "Agent":
            available_identifiers = ['All users'] + sorted(df['identifier'].unique().tolist())
            selected_identifier = st.sidebar.selectbox(
                "👤 User Identifier", 
                available_identifiers,
                help="Select a specific identifier or 'All users'"
            )
            
            # Si un identifiant spécifique est sélectionné, filtrer les données
            if selected_identifier != 'All users':
                df = df[df['identifier'] == selected_identifier]
                if df.empty:
                    st.warning(f"⚠️ No data found for identifier {selected_identifier}.")
                    return
                #st.info(f"ℹ️ Displaying data for user with identifier: {selected_identifier}")
        else:
            # Pour les utilisateurs non-admin et non-agent, l'identifiant est toujours le leur
            selected_identifier = identifier
            #st.info(f"Showing data for your account (identifier: {identifier})")
        
        # Filtres de période pour TOUS les utilisateurs
        available_years = sorted(list(set(df['annee_debut'].tolist() + df['annee_fin'].tolist())))
        if available_years:
            selected_year = st.sidebar.selectbox("📅 Year", available_years, index=len(available_years)-1)
        else:
            st.warning("No data available for the selected filters.")
            return
        
        # Options de période pour TOUS les utilisateurs
        period_type = st.sidebar.radio(
            "⏱️ Period Type",
            ["Annual", "Semi-annual", "Quarterly", "Monthly"]
        )
        
        # Filtres supplémentaires UNIQUEMENT pour admin
        if is_admin:
            # Filtre par ville pour admin
            available_cities = ['All cities'] + sorted(df['Ville'].unique().tolist())
            selected_city = st.sidebar.selectbox("🏙️ City", available_cities)
            
            # Filtres communs pour admin et agents
            client_types = ['All types'] + sorted(df['TypeClient'].unique().tolist())
            selected_client_type = st.sidebar.selectbox("👥 Client Type", client_types)
            
            # Filtre par statut de paiement - UNIQUEMENT POUR ADMIN
            payment_statuses = ['All statuses'] + sorted(df['StatutPaiement'].unique().tolist())
            selected_payment_status = st.sidebar.selectbox("💰 Payment Status", payment_statuses)
        elif user_role == "Agent":
            # Pour les agents, la ville est fixée à leur ville assignée
            selected_city = assigned_city
            
            # Mais ils peuvent filtrer par type de client
            client_types = ['All types'] + sorted(df['TypeClient'].unique().tolist())
            selected_client_type = st.sidebar.selectbox("👥 Client Type", client_types)
            
            # Filtre par statut de paiement pour les agents aussi
            payment_statuses = ['All statuses'] + sorted(df['StatutPaiement'].unique().tolist())
            selected_payment_status = st.sidebar.selectbox("💰 Payment Status", payment_statuses)
        else:
            # Pour les utilisateurs non-admin, utiliser leur ville/type sans afficher de filtres
            if not df.empty:
                selected_city = df['Ville'].iloc[0]
                selected_client_type = df['TypeClient'].iloc[0]
                # Définir selected_payment_status à 'All statuses' par défaut sans afficher le filtre
                selected_payment_status = 'All statuses'
            else:
                selected_city = 'All cities'
                selected_client_type = 'All types'
                selected_payment_status = 'All statuses'
        
        # Appliquer les filtres de base (année)
        df_filtered = df[(df['annee_debut'] == selected_year) | (df['annee_fin'] == selected_year)]
        
        # Filtre par ville
        if selected_city != 'All cities':
            df_filtered = df_filtered[df_filtered['Ville'] == selected_city]
        
        # Filtre par type de client
        if selected_client_type != 'All types':
            df_filtered = df_filtered[df_filtered['TypeClient'] == selected_client_type]
        
        # Filtre par statut de paiement
        if selected_payment_status != 'All statuses':
            df_filtered = df_filtered[df_filtered['StatutPaiement'] == selected_payment_status]
            
        # Filtres additionnels selon le type de période
        if period_type == "Semi-annual":
            selected_semester = st.sidebar.selectbox(
                "🗓️ Semester", 
                [1, 2],
                format_func=lambda x: f"Semester {x}"
            )
            
            # Determine quarters associated with the semester
            if selected_semester == 1:
                semester_quarters = [1, 2]
            else:  # selected_semester == 2
                semester_quarters = [3, 4]
            
            # Apply filter with a stricter condition
            df_filtered = df_filtered[df_filtered['trimestre_debut'].isin(semester_quarters)]
            period_label = f"Semester {selected_semester} {selected_year}"
        
        elif period_type == "Quarterly":
            selected_quarter = st.sidebar.selectbox(
                "📆 Quarter", 
                [1, 2, 3, 4],
                format_func=lambda x: f"Quarter {x}"
            )
            df_filtered = df_filtered[(df_filtered['trimestre_debut'] == selected_quarter) | 
                                (df_filtered['trimestre_fin'] == selected_quarter)]
            period_label = f"Quarter {selected_quarter} {selected_year}"
        
        elif period_type == "Monthly":
            # Month selection
            selected_month = st.sidebar.selectbox(
                "📅 Month", 
                list(range(1, 13)),
                format_func=lambda x: calendar.month_name[x]
            )
            
            # Filter by selected month
            df_filtered = df_filtered[df_filtered['mois_debut'] == selected_month]
            period_label = f"{calendar.month_name[selected_month]} {selected_year}"
        
        else:  # Annual
            period_label = f"Year {selected_year}"
        
        # Check if we have data after filtering
        if df_filtered.empty:
            st.warning("⚠️ No data matches the selected filterss.")
            return
        
        # Display information about applied filters
        st.write(f"### 📈 Data for: {period_label}")
        
        if selected_identifier != 'All users':
            # Display user information
            user_info = df_filtered.iloc[0] if not df_filtered.empty else None
            if user_info is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**👤 Identifier:** {selected_identifier}")
                    st.write(f"**👥 Client Name:** {user_info['ClientName']}")
                with col2:
                    st.write(f"**🏢 Client Type:** {user_info['TypeClient']}")
                    st.write(f"**🏙️ City:** {user_info['Ville']}")
        
        if selected_city != 'All cities' and is_admin:
            st.write(f"**🏙️ City:** {selected_city}")
        
        if selected_client_type != 'All types' and is_admin:
            st.write(f"**🏢 Client Type:** {selected_client_type}")
        
        if selected_payment_status != 'All statuses':
            st.write(f"**💰 Payment Status:** {selected_payment_status}")
        
        # General statistics
        st.write(f"**📋 Number of invoices:** {len(df_filtered)}")
        
        # Create three columns for key statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🚰 Total Consumption", f"{df_filtered['Conso'].sum():,.0f} units")
        
        with col2:
            st.metric("💵 Total Billed Amount", f"{df_filtered['MontFact'].sum():,.0f} FCFA")
        
        with col3:
            st.metric("📊 Average Consumption", f"{df_filtered['Conso'].mean():,.1f} units")
        
        # Analysis by quarter/semester and by city
        st.write("## 🔍 Detailed Analysis")
        
        #Options d'onglets basées sur les permissions
        tab_options = ["📅 By Period"]
                
        # Ajouter les onglets par ville et par type client seulement pour admin
        if is_admin:
            tab_options.extend(["🏙️ By City", "👥 By Client Type"])

        # Créer les onglets
        tabs = st.tabs(tab_options)
                
        with tabs[0]:
            # Data preparation according to period type
            st.write(f"### 📊 Analysis by period")
            if period_type == "Monthly":
                # For monthly analysis, group by week
                df_filtered_with_week = df_filtered[df_filtered['semaine_mois'] <= 5]  # Limit to weeks 1-5
                
                if not df_filtered_with_week.empty:
                    # Group by week of month
                    conso_per_week = df_filtered_with_week.groupby('semaine_mois')['Conso'].sum().reset_index()
                    conso_per_week['semaine_label'] = conso_per_week['semaine_mois'].apply(lambda x: f"Week {x}")
                    
                    # Sort by week to ensure correct order
                    conso_per_week = conso_per_week.sort_values(by='semaine_mois')
                    
                    # Line chart for consumption by week
                    fig_conso = px.line(conso_per_week, x='semaine_label', y='Conso', markers=True,
                            labels={'Conso': 'Total Consumption', 'semaine_label': 'Week'},
                            title=f"🚰 Consumption by week - {period_label}")
                    st.plotly_chart(fig_conso)

                    # Generate and display interpretation for consumption
                    conso_interpretation = generate_interpretation(
                        conso_per_week, 
                        period_type, 
                        grouping_column='semaine_mois', 
                        value_column='Conso',
                        group_label='semaine_label'
                    )
                    st.write(conso_interpretation)
                    
                    # Billed amount by week
                    amount_per_week = df_filtered_with_week.groupby('semaine_mois')['MontFact'].sum().reset_index()
                    amount_per_week['semaine_label'] = amount_per_week['semaine_mois'].apply(lambda x: f"Week {x}")
                    
                    # Sort by week to ensure correct order
                    amount_per_week = amount_per_week.sort_values(by='semaine_mois')
                    
                    fig_amount = px.line(amount_per_week, x='semaine_label', y='MontFact', markers=True,
                                labels={'MontFact': 'Billed Amount (FCFA)', 'semaine_label': 'Week'},
                                title=f"💵 Billed amount by week - {period_label}")
                    st.plotly_chart(fig_amount)


                    # Generate and display interpretation for billing
                    amount_interpretation = generate_interpretation(
                        amount_per_week, 
                        period_type, 
                        grouping_column='semaine_mois', 
                        value_column='MontFact',
                        group_label='semaine_label'
                    )
                    st.write(amount_interpretation)
                else:
                    st.info(f"📌 Not enough weekly data for {period_label}")
            
            elif period_type == "Quarterly":
                # For quarterly analysis, group by month
                # Filtering by month for the selected quarter
                start_month = 3 * (selected_quarter - 1) + 1
                end_month = 3 * selected_quarter
                
                # Group by month
                df_month = df_filtered[df_filtered['mois_debut'].between(start_month, end_month)]
                if not df_month.empty:
                    # Consumption by month
                    conso_per_month = df_month.groupby('mois_debut')['Conso'].sum().reset_index()
                    conso_per_month['nom_mois'] = conso_per_month['mois_debut'].apply(lambda x: calendar.month_abbr[x])
                    
                    # Sort by month to ensure correct order
                    conso_per_month = conso_per_month.sort_values(by='mois_debut')
                    
                    fig_conso = px.line(conso_per_month, x='nom_mois', y='Conso', markers=True,
                            labels={'Conso': 'Total Consumption', 'nom_mois': 'Month'},
                            title=f"🚰 Monthly consumption - Quarter {selected_quarter} {selected_year}")
                    st.plotly_chart(fig_conso)

                    conso_interpretation = generate_interpretation(
                        conso_per_month, 
                        period_type, 
                        grouping_column='mois_debut', 
                        value_column='Conso',
                        group_label='nom_mois'
                    )
                    st.write(conso_interpretation)
                    
                    # Billed amount by month
                    amount_per_month = df_month.groupby('mois_debut')['MontFact'].sum().reset_index()
                    amount_per_month['nom_mois'] = amount_per_month['mois_debut'].apply(lambda x: calendar.month_abbr[x])
                    
                    # Sort by month to ensure correct order
                    amount_per_month = amount_per_month.sort_values(by='mois_debut')
                    
                    fig_amount = px.line(amount_per_month, x='nom_mois', y='MontFact', markers=True,
                                labels={'MontFact': 'Billed Amount (FCFA)', 'nom_mois': 'Month'},
                                title=f"💵 Monthly billed amount - Quarter {selected_quarter} {selected_year}")
                    st.plotly_chart(fig_amount)

                    amount_interpretation = generate_interpretation(
                        amount_per_month, 
                        period_type, 
                        grouping_column='mois_debut', 
                        value_column='MontFact',
                        group_label='nom_mois'
                    )
                    st.write(amount_interpretation)
                else:
                    st.info(f"📌 Not enough monthly data for quarter {selected_quarter}")
            
            elif period_type == "Semi-annual":
                # For semi-annual analysis, group by quarter
                # Determine quarters for the semester
                if selected_semester == 1:
                    quarters = [1, 2]
                else:
                    quarters = [3, 4]
                
                # Filter and group by quarter
                df_quarter = df_filtered[df_filtered['trimestre_debut'].isin(quarters)]
                if not df_quarter.empty:
                    # Consumption by quarter
                    conso_per_quarter = df_quarter.groupby('trimestre_debut')['Conso'].sum().reset_index()
                    conso_per_quarter['trimestre_label'] = conso_per_quarter['trimestre_debut'].apply(get_quarter_label)
                    
                    # Sort by quarter to ensure correct order
                    conso_per_quarter = conso_per_quarter.sort_values(by='trimestre_debut')
                    
                    fig_conso = px.line(conso_per_quarter, x='trimestre_label', y='Conso', markers=True,
                            labels={'Conso': 'Total Consumption', 'trimestre_label': 'Quarter'},
                            title=f"🚰 Quarterly consumption - Semester {selected_semester} {selected_year}")
                    st.plotly_chart(fig_conso)

                    # Generate and display interpretation for consumption
                    conso_interpretation = generate_interpretation(
                        conso_per_quarter, 
                        period_type, 
                        grouping_column='trimestre_debut', 
                        value_column='Conso',
                        group_label='trimestre_label'
                    )
                    st.write(conso_interpretation)


                    
                    # Billed amount by quarter
                    amount_per_quarter = df_quarter.groupby('trimestre_debut')['MontFact'].sum().reset_index()
                    amount_per_quarter['trimestre_label'] = amount_per_quarter['trimestre_debut'].apply(get_quarter_label)
                    
                    # Sort by quarter to ensure correct order
                    amount_per_quarter = amount_per_quarter.sort_values(by='trimestre_debut')
                    
                    fig_amount = px.line(amount_per_quarter, x='trimestre_label', y='MontFact', markers=True,
                                labels={'MontFact': 'Billed Amount (FCFA)', 'trimestre_label': 'Quarter'},
                                title=f"💵 Quarterly billed amount - Semester {selected_semester} {selected_year}")
                    st.plotly_chart(fig_amount)


                    # Generate and display interpretation for billing
                    amount_interpretation = generate_interpretation(
                        amount_per_quarter, 
                        period_type, 
                        grouping_column='trimestre_debut', 
                        value_column='MontFact',
                        group_label='trimestre_label'
                    )
                    st.write(amount_interpretation)
                else:
                    st.info(f"📌 Not enough quarterly data for semester {selected_semester}")
            
            else:  # Annual
                # For annual analysis, group by semester
                conso_per_semester = df_filtered.groupby('semestre_debut')['Conso'].sum().reset_index()
                
                # Use get_semester_label instead of get_quarter_label
                conso_per_semester['semestre_label'] = conso_per_semester['semestre_debut'].apply(get_semester_label)
                
                # Sort by semester to ensure correct order
                conso_per_semester = conso_per_semester.sort_values(by='semestre_debut')
                
                fig_conso = px.line(conso_per_semester, x='semestre_label', y='Conso', markers=True,
                        labels={'Conso': 'Total Consumption', 'semestre_label': 'Semester'},
                        title=f"🚰 Semi-annual consumption - {selected_year}")
                st.plotly_chart(fig_conso)

                # Generate and display interpretation for consumption
                conso_interpretation = generate_interpretation(
                    conso_per_semester, 
                    period_type, 
                    grouping_column='semestre_debut', 
                    value_column='Conso',
                    group_label='semestre_label'
                )
                st.write(conso_interpretation)
                
                # Billed amount by semester
                amount_per_semester = df_filtered.groupby('semestre_debut')['MontFact'].sum().reset_index()
                
                # Use get_semester_label instead of get_quarter_label
                amount_per_semester['semestre_label'] = amount_per_semester['semestre_debut'].apply(get_semester_label)
                
                # Sort by semester to ensure correct order
                amount_per_semester = amount_per_semester.sort_values(by='semestre_debut')
                
                fig_amount = px.line(amount_per_semester, x='semestre_label', y='MontFact', markers=True,
                            labels={'MontFact': 'Billed Amount (FCFA)', 'semestre_label': 'Semester'},
                            title=f"💵 Semi-annual billed amount - {selected_year}")
                st.plotly_chart(fig_amount)

                # Generate and display interpretation for billing
                amount_interpretation = generate_interpretation(
                    amount_per_semester, 
                    period_type, 
                    grouping_column='semestre_debut', 
                    value_column='MontFact',
                    group_label='semestre_label'
                )
                st.write(amount_interpretation)
        
        if is_admin:
            city_tab_index = 1 
            with tabs[city_tab_index]:
            # st.write("### 🏙️ Analysis by city")
                
                # Check if filter is already applied to a single city
                if selected_city == 'All cities':
                    st.write(f"### 📊 Analysis by city")
                    # Group by city
                    conso_per_city = df_filtered.groupby('Ville')['Conso'].sum().reset_index()
                    
                    # Bar chart for consumption by city
                    fig_conso = px.bar(conso_per_city, x='Ville', y='Conso',
                            labels={'Conso': 'Total Consumption', 'Ville': 'City'},
                            title="🚰 Total consumption by city",
                            color='Ville')
                    st.plotly_chart(fig_conso)

                    # Generate and display city comparison interpretation
                    if len(conso_per_city) > 1:
                        max_city = conso_per_city.loc[conso_per_city['Conso'].idxmax()]['Ville']
                        min_city = conso_per_city.loc[conso_per_city['Conso'].idxmin()]['Ville']
                        max_conso = conso_per_city['Conso'].max()
                        min_conso = conso_per_city['Conso'].min()
                        total_conso = conso_per_city['Conso'].sum()
                        
                        city_interpretation = f"""
                        **📊 Interpretation:** The total consumption across all cities is {total_conso:,.1f} units.
                        {max_city} has the highest consumption ({max_conso:,.1f} units or {(max_conso/total_conso)*100:.1f}% of total),
                        while {min_city} has the lowest ({min_conso:,.1f} units or {(min_conso/total_conso)*100:.1f}% of total).
                        """
                        
                        if (max_conso / min_conso) > 5:
                            city_interpretation += f" There is a very significant disparity between cities, with {max_city} consuming {(max_conso/min_conso):.1f} times more water than {min_city}."
                        elif (max_conso / min_conso) > 2:
                            city_interpretation += f" There is a notable difference between cities, with {max_city} consuming {(max_conso/min_conso):.1f} times more water than {min_city}."
                        else:
                            city_interpretation += " The consumption is relatively balanced across the different cities."
                            
                        st.write(city_interpretation)
                    
                    # Bar chart for billed amounts by city
                    amount_per_city = df_filtered.groupby('Ville')['MontFact'].sum().reset_index()
                    fig_amount = px.bar(amount_per_city, x='Ville', y='MontFact',
                                labels={'MontFact': 'Billed Amount (FCFA)', 'Ville': 'City'},
                                title="💵 Total billed amount by city",
                                color='Ville')
                    st.plotly_chart(fig_amount)


                    # Generate and display billing city comparison interpretation
                    if len(amount_per_city) > 1:
                        max_city = amount_per_city.loc[amount_per_city['MontFact'].idxmax()]['Ville']
                        min_city = amount_per_city.loc[amount_per_city['MontFact'].idxmin()]['Ville']
                        max_amount = amount_per_city['MontFact'].max()
                        min_amount = amount_per_city['MontFact'].min()
                        total_amount = amount_per_city['MontFact'].sum()
                        
                        billing_interpretation = f"""
                        **📊 Interpretation:** The total billing across all cities is {total_amount:,.1f} FCFA.
                        {max_city} has the highest billing amount ({max_amount:,.1f} FCFA or {(max_amount/total_amount)*100:.1f}% of total),
                        while {min_city} has the lowest ({min_amount:,.1f} FCFA or {(min_amount/total_amount)*100:.1f}% of total).
                        """
                        
                        if (max_amount / min_amount) > 5:
                            billing_interpretation += f" There is a very significant disparity in billing between cities, with {max_city} generating {(max_amount/min_amount):.1f} times more revenue than {min_city}."
                        elif (max_amount / min_amount) > 2:
                            billing_interpretation += f" There is a notable difference in billing between cities, with {max_city} generating {(max_amount/min_amount):.1f} times more revenue than {min_city}."
                        else:
                            billing_interpretation += " The billing is relatively balanced across the different cities."
                            
                        st.write(billing_interpretation)
                else:
                    # For a specific city, show evolution over time
                    st.write(f"### 📊 Analysis in {selected_city}")
                    
                    # Depending on period type, group differently
                    if period_type == "Monthly":
                        # Group by week for the selected month
                        df_week = df_filtered[df_filtered['semaine_mois'] <= 5]
                        if not df_week.empty:
                            # Consumption by week
                            conso_per_week = df_week.groupby('semaine_mois')['Conso'].sum().reset_index()
                            conso_per_week['semaine_label'] = conso_per_week['semaine_mois'].apply(lambda x: f"Week {x}")
                            
                            # Sort by week
                            conso_per_week = conso_per_week.sort_values(by='semaine_mois')
                            
                            fig_conso = px.bar(conso_per_week, x='semaine_label', y='Conso',
                                    labels={'Conso': 'Consumption', 'semaine_label': 'Week'},
                                    title=f"🚰 Weekly consumption in {selected_city} - {period_label}")
                            st.plotly_chart(fig_conso)

                            # Generate and display interpretation for consumption
                            conso_interpretation = generate_interpretation(
                                conso_per_week, 
                                period_type, 
                                grouping_column='semaine_mois', 
                                value_column='Conso',
                                group_label='semaine_label'
                            )
                            st.write(conso_interpretation)
                            
                            # Billed amount by week
                            amount_per_week = df_week.groupby('semaine_mois')['MontFact'].sum().reset_index()
                            amount_per_week['semaine_label'] = amount_per_week['semaine_mois'].apply(lambda x: f"Week {x}")
                            
                            # Sort by week
                            amount_per_week = amount_per_week.sort_values(by='semaine_mois')
                            
                            fig_amount = px.bar(amount_per_week, x='semaine_label', y='MontFact',
                                        labels={'MontFact': 'Amount (FCFA)', 'semaine_label': 'Week'},
                                        title=f"💵 Weekly billed amount in {selected_city} - {period_label}")
                            st.plotly_chart(fig_amount)

                            # Generate and display interpretation for billing
                            amount_interpretation = generate_interpretation(
                                amount_per_week, 
                                period_type, 
                                grouping_column='semaine_mois', 
                                value_column='MontFact',
                                group_label='semaine_label'
                            )
                            st.write(amount_interpretation)
                        else:
                            st.info(f"📌 Not enough weekly data for {selected_city} in {period_label}")
                    
                    elif period_type == "Annual":
                        # Group by semester for the year
                        conso_per_semester = df_filtered.groupby('semestre_debut')['Conso'].sum().reset_index()
                        conso_per_semester['semestre_label'] = conso_per_semester['semestre_debut'].apply(get_semester_label)
                        
                        # Sort by semester
                        conso_per_semester = conso_per_semester.sort_values(by='semestre_debut')
                        
                        fig_conso = px.bar(conso_per_semester, x='semestre_label', y='Conso',
                                labels={'Conso': 'Consumption', 'semestre_label': 'Semester'},
                                title=f"🚰 Semi-annual consumption in {selected_city} - {period_label}")
                        st.plotly_chart(fig_conso)
                        

                        # Generate and display interpretation for consumption
                        conso_interpretation = generate_interpretation(
                            conso_per_semester, 
                            period_type, 
                            grouping_column='semestre_debut', 
                            value_column='Conso',
                            group_label='semestre_label'
                        )
                        st.write(conso_interpretation)


                        # Billed amount by semester
                        amount_per_semester = df_filtered.groupby('semestre_debut')['MontFact'].sum().reset_index()
                        amount_per_semester['semestre_label'] = amount_per_semester['semestre_debut'].apply(get_semester_label)
                        
                        # Sort by semester
                        amount_per_semester = amount_per_semester.sort_values(by='semestre_debut')
                        
                        fig_amount = px.bar(amount_per_semester, x='semestre_label', y='MontFact',
                                    labels={'MontFact': 'Amount (FCFA)', 'semestre_label': 'Semester'},
                                    title=f"💵 Semi-annual billed amount in {selected_city} - {period_label}")
                        st.plotly_chart(fig_amount)

                        # Generate and display interpretation for billing
                        amount_interpretation = generate_interpretation(
                            amount_per_semester, 
                            period_type, 
                            grouping_column='semestre_debut', 
                            value_column='MontFact',
                            group_label='semestre_label'
                        )
                        st.write(amount_interpretation)
                    
                    else:  # Semi-annual or Quarterly
                        # Group based on period
                        if period_type == "Quarterly":
                            # Group by month for the quarter
                            conso_per_period = df_filtered.groupby('mois_debut')['Conso'].sum().reset_index()
                            conso_per_period['periode_label'] = conso_per_period['mois_debut'].apply(lambda x: calendar.month_abbr[x])
                            x_label = "Month"
                            group_column = 'mois_debut'
                            
                            # Sort by month
                            conso_per_period = conso_per_period.sort_values(by='mois_debut')
                        else:  # Semi-annual
                            # Group by quarter for the semester
                            conso_per_period = df_filtered.groupby('trimestre_debut')['Conso'].sum().reset_index()
                            conso_per_period['periode_label'] = conso_per_period['trimestre_debut'].apply(get_quarter_label)
                            x_label = "Quarter"
                            group_column = 'trimestre_debut'
                            
                            # Sort by quarter
                            conso_per_period = conso_per_period.sort_values(by='trimestre_debut')
                        
                        # Consumption by period
                        fig_conso = px.bar(conso_per_period, x='periode_label', y='Conso',
                                labels={'Conso': 'Consumption', 'periode_label': x_label},
                                title=f"🚰 Consumption in {selected_city} - {period_label}")
                        st.plotly_chart(fig_conso)

                        # Generate and display interpretation for consumption
                        conso_interpretation = generate_interpretation(
                            conso_per_period, 
                            period_type, 
                            grouping_column=group_column, 
                            value_column='Conso',
                            group_label='periode_label'
                        )
                        st.write(conso_interpretation)

                        
                        # Billed amount by period
                        if period_type == "Quarterly":
                            amount_per_period = df_filtered.groupby('mois_debut')['MontFact'].sum().reset_index()
                            amount_per_period['periode_label'] = amount_per_period['mois_debut'].apply(lambda x: calendar.month_abbr[x])
                            # Sort by month
                            amount_per_period = amount_per_period.sort_values(by='mois_debut')
                        else:  # Semi-annual
                            amount_per_period = df_filtered.groupby('trimestre_debut')['MontFact'].sum().reset_index()
                            amount_per_period['periode_label'] = amount_per_period['trimestre_debut'].apply(get_quarter_label)
                            # Sort by quarter
                            amount_per_period = amount_per_period.sort_values(by='trimestre_debut')
                        
                        fig_amount = px.bar(amount_per_period, x='periode_label', y='MontFact',
                                    labels={'MontFact': 'Amount (FCFA)', 'periode_label': x_label},
                                    title=f"💵 Billed amount in {selected_city} - {period_label}")
                        st.plotly_chart(fig_amount)

                        # Generate and display interpretation for billing
                        amount_interpretation = generate_interpretation(
                            amount_per_period, 
                            period_type, 
                            grouping_column=group_column, 
                            value_column='MontFact',
                            group_label='periode_label'
                        )
                        st.write(amount_interpretation)

            client_tab_index = 2
            with tabs[client_tab_index]:


                # Analyse par type de client    
                # Check if a client type is already selected
                if selected_client_type == 'All types':
                    st.write("### 📊 Analysis by client type")
                    # Group by client type
                    conso_per_type = df_filtered.groupby('TypeClient')['Conso'].sum().reset_index()
                    
                    # Pie chart for consumption by client type
                    fig_conso = px.pie(conso_per_type, values='Conso', names='TypeClient',
                            title="🚰 Distribution of consumption by client type")
                    st.plotly_chart(fig_conso)

                    # Generate client type consumption interpretation
                    if len(conso_per_type) > 1:
                        top_client = conso_per_type.loc[conso_per_type['Conso'].idxmax()]['TypeClient']
                        top_consumption = conso_per_type['Conso'].max()
                        total_consumption = conso_per_type['Conso'].sum()
                        top_percentage = (top_consumption / total_consumption) * 100
                        
                        client_interpretation = f"""
                        **📊 Interpretation:** Among the different client types, {top_client} accounts for the largest portion 
                        of water consumption at {top_percentage:.1f}% of the total ({top_consumption:,.1f} units).
                        """
                        
                        # Add additional analysis if there are several client types
                        if len(conso_per_type) > 2:
                            second_client = conso_per_type.sort_values(by='Conso', ascending=False).iloc[1]['TypeClient']
                            second_consumption = conso_per_type.sort_values(by='Conso', ascending=False).iloc[1]['Conso']
                            second_percentage = (second_consumption / total_consumption) * 100
                            
                            client_interpretation += f" This is followed by {second_client} at {second_percentage:.1f}% ({second_consumption:,.1f} units)."
                            
                            if top_percentage > 60:
                                client_interpretation += f" The {top_client} client type dominates water consumption in this period."
                            elif top_percentage > 40:
                                client_interpretation += f" The {top_client} client type represents a significant portion of water consumption."
                            else:
                                client_interpretation += " The consumption is relatively distributed among different client types."
                        
                        st.write(client_interpretation)
                    
                    # Pie chart for billed amounts by client type
                    amount_per_type = df_filtered.groupby('TypeClient')['MontFact'].sum().reset_index()
                    fig_amount = px.pie(amount_per_type, values='MontFact', names='TypeClient',
                                title="💵 Distribution of billed amounts by client type")
                    st.plotly_chart(fig_amount)

                    # Generate client type billing interpretation
                    if len(amount_per_type) > 1:
                        top_client = amount_per_type.loc[amount_per_type['MontFact'].idxmax()]['TypeClient']
                        top_amount = amount_per_type['MontFact'].max()
                        total_amount = amount_per_type['MontFact'].sum()
                        top_percentage = (top_amount / total_amount) * 100
                        
                        billing_interpretation = f"""
                        **📊 Interpretation:** Among the different client types, {top_client} accounts for the largest portion 
                        of billing revenue at {top_percentage:.1f}% of the total ({top_amount:,.1f} FCFA).
                        """
                        
                        # Add additional analysis if there are several client types
                        if len(amount_per_type) > 2:
                            second_client = amount_per_type.sort_values(by='MontFact', ascending=False).iloc[1]['TypeClient']
                            second_amount = amount_per_type.sort_values(by='MontFact', ascending=False).iloc[1]['MontFact']
                            second_percentage = (second_amount / total_amount) * 100
                            
                            billing_interpretation += f" This is followed by {second_client} at {second_percentage:.1f}% ({second_amount:,.1f} FCFA)."
                            
                            if top_percentage > 60:
                                billing_interpretation += f" The {top_client} client type dominates billing revenue in this period."
                            elif top_percentage > 40:
                                billing_interpretation += f" The {top_client} client type represents a significant portion of billing revenue."
                            else:
                                billing_interpretation += " The billing revenue is relatively distributed among different client types."
                        
                        # Compare billing to consumption patterns if both exist
                        if len(conso_per_type) > 1 and len(amount_per_type) > 1:
                            top_conso_client = conso_per_type.loc[conso_per_type['Conso'].idxmax()]['TypeClient']
                            if top_client != top_conso_client:
                                billing_interpretation += f"\n\nInterestingly, while {top_client} generates the most revenue, {top_conso_client} consumes the most water, which might indicate different pricing structures or consumption patterns across client types."
                        
                        st.write(billing_interpretation)

                else:
                    # Detailed analysis for a specific client type
                    
                    
                    # Distribution by city for this client type
                    if selected_city == 'All cities':
                        st.write(f"### 📊 Detailed analysis for clients of type {selected_client_type}")
                        # Consumption by city for this client type
                        conso_per_city = df_filtered.groupby('Ville')['Conso'].sum().reset_index()
                        fig_conso = px.pie(conso_per_city, values='Conso', names='Ville',
                                title=f"🚰 Distribution of consumption by city for {selected_client_type}")
                        st.plotly_chart(fig_conso)


                        # Generate city distribution interpretation for specific client type
                        if len(conso_per_city) > 1:
                            top_city = conso_per_city.loc[conso_per_city['Conso'].idxmax()]['Ville']
                            top_consumption = conso_per_city['Conso'].max()
                            total_consumption = conso_per_city['Conso'].sum()
                            top_percentage = (top_consumption / total_consumption) * 100
                            
                            city_client_interpretation = f"""
                            **📊 Interpretation:** For {selected_client_type} clients, {top_city} accounts for {top_percentage:.1f}% 
                            of the total consumption ({top_consumption:,.1f} units out of {total_consumption:,.1f} units).
                            """
                            
                            if len(conso_per_city) > 2:
                                second_city = conso_per_city.sort_values(by='Conso', ascending=False).iloc[1]['Ville']
                                second_consumption = conso_per_city.sort_values(by='Conso', ascending=False).iloc[1]['Conso']
                                second_percentage = (second_consumption / total_consumption) * 100
                                
                                city_client_interpretation += f" The second highest consumption for this client type is in {second_city} at {second_percentage:.1f}% ({second_consumption:,.1f} units)."
                            
                            if top_percentage > 70:
                                city_client_interpretation += f" {selected_client_type} clients in {top_city} significantly dominate the consumption pattern."
                            elif top_percentage > 50:
                                city_client_interpretation += f" {selected_client_type} clients in {top_city} represent a majority of the consumption."
                            elif top_percentage > 30:
                                city_client_interpretation += f" {selected_client_type} clients in {top_city} represent a substantial portion of the consumption."
                            else:
                                city_client_interpretation += f" {selected_client_type} clients' consumption is relatively distributed across multiple cities."
                            
                            st.write(city_client_interpretation)
                        
                        # Billed amount by city for this client type
                        amount_per_city = df_filtered.groupby('Ville')['MontFact'].sum().reset_index()
                        fig_amount = px.pie(amount_per_city, values='MontFact', names='Ville',
                                title=f"💵 Distribution of billed amounts by city for {selected_client_type}")
                        st.plotly_chart(fig_amount)

                        # Generate billing distribution interpretation for specific client type
                        if len(amount_per_city) > 1:
                            top_city = amount_per_city.loc[amount_per_city['MontFact'].idxmax()]['Ville']
                            top_amount = amount_per_city['MontFact'].max()
                            total_amount = amount_per_city['MontFact'].sum()
                            top_percentage = (top_amount / total_amount) * 100
                            
                            city_billing_interpretation = f"""
                            **📊 Interpretation:** For {selected_client_type} clients, {top_city} accounts for {top_percentage:.1f}% 
                            of the total billing ({top_amount:,.1f} FCFA out of {total_amount:,.1f} FCFA).
                            """
                            
                            if len(amount_per_city) > 2:
                                second_city = amount_per_city.sort_values(by='MontFact', ascending=False).iloc[1]['Ville']
                                second_amount = amount_per_city.sort_values(by='MontFact', ascending=False).iloc[1]['MontFact']
                                second_percentage = (second_amount / total_amount) * 100
                                
                                city_billing_interpretation += f" The second highest billing for this client type is in {second_city} at {second_percentage:.1f}% ({second_amount:,.1f} FCFA)."
                            
                            if top_percentage > 70:
                                city_billing_interpretation += f" {selected_client_type} clients in {top_city} significantly dominate the billing pattern."
                            elif top_percentage > 50:
                                city_billing_interpretation += f" {selected_client_type} clients in {top_city} represent a majority of the billing."
                            elif top_percentage > 30:
                                city_billing_interpretation += f" {selected_client_type} clients in {top_city} represent a substantial portion of the billing."
                            else:
                                city_billing_interpretation += f" {selected_client_type} clients' billing is relatively distributed across multiple cities."
                            
                            # Compare consumption and billing patterns
                            if len(conso_per_city) > 1:
                                top_conso_city = conso_per_city.loc[conso_per_city['Conso'].idxmax()]['Ville']
                                if top_city != top_conso_city:
                                    city_billing_interpretation += f"\n\nIt's noteworthy that while {top_city} generates the most revenue, {top_conso_city} has the highest consumption for {selected_client_type} clients, which might indicate different pricing structures or consumption efficiencies between cities."
                            
                            st.write(city_billing_interpretation)
                    else:
                        st.write(f"### 📊 Detailed analysis for clients of type {selected_client_type} in {selected_city}")
                        # If a specific city is selected, show evolution over time
                        # Select grouping according to period
                        if period_type == "Monthly":
                            # Group by week
                            conso_per_period = df_filtered.groupby('semaine_mois')['Conso'].sum().reset_index()
                            conso_per_period['periode_label'] = conso_per_period['semaine_mois'].apply(lambda x: f"Week {x}")
                            x_label = "Week"
                            group_column = 'semaine_mois'
                            # Sort by week
                            conso_per_period = conso_per_period.sort_values(by='semaine_mois')
                        elif period_type == "Quarterly":
                            # Group by month
                            conso_per_period = df_filtered.groupby('mois_debut')['Conso'].sum().reset_index()
                            conso_per_period['periode_label'] = conso_per_period['mois_debut'].apply(lambda x: calendar.month_abbr[x])
                            x_label = "Month"
                            group_column = 'mois_debut'
                            # Sort by month
                            conso_per_period = conso_per_period.sort_values(by='mois_debut')
                            
                        elif period_type == "Semi-annual": # 📆
                            # Group by quarter 📊
                            conso_per_period = df_filtered.groupby('trimestre_debut')['Conso'].sum().reset_index()
                            conso_per_period['periode_label'] = conso_per_period['trimestre_debut'].apply(get_quarter_label)
                            x_label = "Quarter"
                            group_column = 'trimestre_debut'
                            # Sort by quarter 🔄
                            conso_per_period = conso_per_period.sort_values(by='trimestre_debut')
                        else:  # Annual 📅
                            # Group by semester 📈
                            conso_per_period = df_filtered.groupby('semestre_debut')['Conso'].sum().reset_index()
                            conso_per_period['periode_label'] = conso_per_period['semestre_debut'].apply(get_semester_label)
                            x_label = "Semester"
                            group_column = 'semestre_debut'

                            # Sort by semester 🔄
                            conso_per_period = conso_per_period.sort_values(by='semestre_debut')
                        
                        # No graph if no data after filtering 🔍
                        if not conso_per_period.empty:
                            # Pie chart for consumption distribution by period 🥧
                            fig_conso = px.pie(conso_per_period, values='Conso', names='periode_label',
                                    title=f"🚰 Consumption distribution by {x_label.lower()} for {selected_client_type} in {selected_city}")
                            st.plotly_chart(fig_conso)


                            # Generate time-based interpretation for specific client and city
                            if len(conso_per_period) > 1:
                                top_period = conso_per_period.loc[conso_per_period['Conso'].idxmax()]['periode_label']
                                top_consumption = conso_per_period['Conso'].max()
                                total_consumption = conso_per_period['Conso'].sum()
                                top_percentage = (top_consumption / total_consumption) * 100
                                
                                period_interpretation = f"""
                                **📊 Interpretation:** For {selected_client_type} clients in {selected_city}, {top_period} accounts for {top_percentage:.1f}% 
                                of the total consumption ({top_consumption:,.1f} units out of {total_consumption:,.1f} units).
                                """
                                
                                # Add trend analysis for time-based data
                                if len(conso_per_period) > 2:
                                    periods = conso_per_period['periode_label'].tolist()
                                    first_period = periods[0]
                                    last_period = periods[-1]
                                    first_value = conso_per_period.iloc[0]['Conso']
                                    last_value = conso_per_period.iloc[-1]['Conso']
                                    
                                    # Calculate trend
                                    if last_value > first_value:
                                        pct_change = ((last_value - first_value) / first_value * 100) if first_value > 0 else 100
                                        period_interpretation += f"\n\nFrom {first_period} to {last_period}, consumption has increased by {pct_change:.1f}%, "
                                        period_interpretation += "showing an upward trend in water usage for this client type in this location."
                                    elif last_value < first_value:
                                        pct_change = ((first_value - last_value) / first_value * 100) if first_value > 0 else 100
                                        period_interpretation += f"\n\nFrom {first_period} to {last_period}, consumption has decreased by {pct_change:.1f}%, "
                                        period_interpretation += "showing a downward trend in water usage for this client type in this location."
                                    else:
                                        period_interpretation += f"\n\nConsumption levels remained stable between {first_period} and {last_period} for this client type in this location."
                                
                                st.write(period_interpretation)
                            
                            # Billed amount by period 💰
                            # Similar structure to consumption grouping above
                            if period_type == "Monthly": # 📅
                                amount_per_period = df_filtered.groupby('semaine_mois')['MontFact'].sum().reset_index()
                                amount_per_period['periode_label'] = amount_per_period['semaine_mois'].apply(lambda x: f"Week {x}")
                                amount_per_period = amount_per_period.sort_values(by='semaine_mois')
                            elif period_type == "Quarterly": # 📊
                                amount_per_period = df_filtered.groupby('mois_debut')['MontFact'].sum().reset_index()
                                amount_per_period['periode_label'] = amount_per_period['mois_debut'].apply(lambda x: calendar.month_abbr[x])
                                amount_per_period = amount_per_period.sort_values(by='mois_debut')
                            elif period_type == "Semi-annual": # 📆
                                amount_per_period = df_filtered.groupby('trimestre_debut')['MontFact'].sum().reset_index()
                                amount_per_period['periode_label'] = amount_per_period['trimestre_debut'].apply(get_quarter_label)
                                amount_per_period = amount_per_period.sort_values(by='trimestre_debut')
                            else:  # Annual 📅
                                amount_per_period = df_filtered.groupby('semestre_debut')['MontFact'].sum().reset_index()
                                amount_per_period['periode_label'] = amount_per_period['semestre_debut'].apply(get_semester_label)
                                amount_per_period = amount_per_period.sort_values(by='semestre_debut')
                            
                            fig_amount = px.pie(amount_per_period, values='MontFact', names='periode_label',
                                    title=f"💵 Billed amount distribution by {x_label.lower()} for {selected_client_type} in {selected_city}")
                            st.plotly_chart(fig_amount)

                            # Generate billing time-based interpretation
                            if len(amount_per_period) > 1:
                                top_period = amount_per_period.loc[amount_per_period['MontFact'].idxmax()]['periode_label']
                                top_amount = amount_per_period['MontFact'].max()
                                total_amount = amount_per_period['MontFact'].sum()
                                top_percentage = (top_amount / total_amount) * 100
                                
                                amount_interpretation = f"""
                                **📊 Interpretation:** For {selected_client_type} clients in {selected_city}, {top_period} accounts for {top_percentage:.1f}% 
                                of the total billing ({top_amount:,.1f} FCFA out of {total_amount:,.1f} FCFA).
                                """
                                
                                # Compare with consumption pattern
                                if len(conso_per_period) > 1:
                                    top_conso_period = conso_per_period.loc[conso_per_period['Conso'].idxmax()]['periode_label']
                                    if top_period != top_conso_period:
                                        amount_interpretation += f"\n\nInterestingly, while the highest billing occurs in {top_period}, the highest consumption is in {top_conso_period}, which might indicate seasonal variations in pricing or different consumption patterns."
                                
                                # Add trend analysis for time-based billing data
                                if len(amount_per_period) > 2:
                                    periods = amount_per_period['periode_label'].tolist()
                                    first_period = periods[0]
                                    last_period = periods[-1]
                                    first_value = amount_per_period.iloc[0]['MontFact']
                                    last_value = amount_per_period.iloc[-1]['MontFact']
                                    
                                    # Calculate trend
                                    if last_value > first_value:
                                        pct_change = ((last_value - first_value) / first_value * 100) if first_value > 0 else 100
                                        amount_interpretation += f"\n\nFrom {first_period} to {last_period}, billing has increased by {pct_change:.1f}%, "
                                        amount_interpretation += "showing an upward trend in revenue from this client type in this location."
                                    elif last_value < first_value:
                                        pct_change = ((first_value - last_value) / first_value * 100) if first_value > 0 else 100
                                        amount_interpretation += f"\n\nFrom {first_period} to {last_period}, billing has decreased by {pct_change:.1f}%, "
                                        amount_interpretation += "showing a downward trend in revenue from this client type in this location."
                                    else:
                                        amount_interpretation += f"\n\nBilling amounts remained stable between {first_period} and {last_period} for this client type in this location."
                                
                                st.write(amount_interpretation)
                        else:
                            st.info(f"📌 Not enough data for {selected_client_type} in {selected_city} for the selected period. Try adjusting your filters! 🔍")
            


            
            
                        # Dynamic table showing the number of clients by type before the client type analysis title
                # Count unique clients by type
                client_count = df.groupby('TypeClient')['identifier'].nunique().reset_index()
                client_count.columns = ['Client Type', 'Number of Clients']
                
                # Add a total row
                total_clients = client_count['Number of Clients'].sum()
                total_row = pd.DataFrame({'Client Type': ['Total'], 'Number of Clients': [total_clients]})
                client_count = pd.concat([client_count, total_row], ignore_index=True)
                
                # Display the table with a title
                st.write("### 🗂️📌 Distribution of Clients by Type")
                st.table(client_count.style.set_properties(**{'text-align': 'center'}).format({'Number of Clients': '{:,d}'}))
                
                # Add an interpretation of the table
                if len(client_count) > 2:  # More than one client type + the total row
                    max_type = client_count.iloc[:-1]['Number of Clients'].idxmax()  # Exclude the total row
                    max_count = client_count.iloc[max_type]['Number of Clients']
                    max_pct = (max_count / total_clients) * 100
                    
                    st.write(f"""
                    **Interpretation:** The system has a total of {total_clients:,d} registered clients. 
                    The "{client_count.iloc[max_type]['Client Type']}" type represents the largest category with 
                    {max_count:,d} clients ({max_pct:.1f}% of the total).
                    """)


        # Invoice history for a specific user 📋
        if selected_identifier != 'All users':
            st.write("## 📜 Invoice History")
            
            # Sort invoices by date 📅
            invoices = df_filtered.sort_values(by='periodeDebut', ascending=False)
            
            # Display a table of invoices 📊
            if not invoices.empty:
                # Select relevant columns 🔍
                invoice_columns = ['id_facture', 'periodeDebut', 'periodeFin', 'Conso', 'MontFact', 'StatutPaiement']
                
                # Format dates 📆
                invoices_display = invoices[invoice_columns].copy()
                invoices_display['periodeDebut'] = invoices_display['periodeDebut'].dt.strftime('%d/%m/%Y')
                invoices_display['periodeFin'] = invoices_display['periodeFin'].dt.strftime('%d/%m/%Y')
                
                # Rename columns for display 🏷️
                invoices_display.columns = ['Invoice ID', 'Start Period', 'End Period', 'Consumption', 'Amount (FCFA)', 'Status']
                
                st.dataframe(invoices_display, use_container_width=True)
                
                # # Graph of consumption evolution 📈
                # st.write("### 📈 Consumption Evolution")
                
                # fig = px.line(invoices.sort_values(by='periodeDebut'), 
                #              x='periodeDebut', 
                #              y='Conso',
                #              markers=True,
                #              labels={'periodeDebut': 'Date', 'Conso': 'Consumption (units)'},
                #              title=f"🚰 Consumption evolution for user {selected_identifier}")
                # st.plotly_chart(fig)
                
                #  # Graph of billed amounts 💵
                # st.write("### 💵 Billed Amounts")
                
                # fig = px.line(invoices.sort_values(by='periodeDebut'), 
                #             x='periodeDebut', 
                #             y='MontFact',
                #             markers=True,
                #             labels={'periodeDebut': 'Date', 'MontFact': 'Amount (FCFA)'},
                #             title=f"💰 Evolution of billed amounts for user {selected_identifier}")
                # st.plotly_chart(fig)
        
        # Download filtered data 💾
        st.write("## 📥 Download Filtered Data")
        
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📋 Download as CSV",
            data=csv,
            file_name=f"sodeci_consumption_{period_label.replace(' ', '_')}.csv",
            mime="text/csv",
        )
        
        # Display raw data (optional, can be commented if not desired) 🔎
        with st.expander("👁️ View Raw Data"):
            st.dataframe(df_filtered)

    # else:
    #     st.error("Please Login first")

if __name__ == "__main__":
    dashboard()  

    

