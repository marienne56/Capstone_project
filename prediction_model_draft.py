
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import calendar
from datetime import datetime, timedelta
from dbConnection import get_connection  # Utilise votre fonction de connexion existante







# Fonction pour obtenir le trimestre à partir d'une date
def get_quarter(date):
    """Returns the quarter (1-4) for a given date"""
    return (pd.to_datetime(date).month - 1) // 3 + 1

# Fonction pour obtenir le semestre à partir d'une date
def get_semester(date):
    """Returns the semester (1-2) for a given date"""
    return 1 if pd.to_datetime(date).month <= 6 else 2

# Fonction pour convertir un numéro de trimestre en étiquette
def get_quarter_label(quarter):
    """Converts a quarter number to label"""
    return f"Q{quarter}"

# Fonction pour convertir un numéro de semestre en étiquette
def get_semester_label(semester):
    """Converts a semester number to label"""
    return f"S{semester}"

# Fonction pour calculer la semaine du mois
def get_week_of_month(date):
    """Calculates the week of the month (1-5) for a given date"""
    date = pd.to_datetime(date)
    first_day = date.replace(day=1)
    dom = date.day
    adjusted_dom = dom + first_day.weekday()
    return (adjusted_dom - 1) // 7 + 1

# Fonction pour le chargement et la préparation des données
@st.cache_data
def load_prediction_data():
    try:
        engine = get_connection()
        
        # Exécuter une requête SQL pour récupérer toutes les données de la table consumption
        query = "SELECT * FROM consumption"
        df = pd.read_sql(query, engine)
        
        # Conversion des colonnes de date
        df['periodeDebut'] = pd.to_datetime(df['periodeDebut'])
        df['periodeFin'] = pd.to_datetime(df['periodeFin'])
        
        # Ajout des colonnes d'analyse temporelle
        df['annee_debut'] = df['periodeDebut'].dt.year
        df['trimestre_debut'] = df['periodeDebut'].apply(get_quarter)
        df['semestre_debut'] = df['periodeDebut'].apply(get_semester)
        df['mois_debut'] = df['periodeDebut'].dt.month
        df['nom_mois_debut'] = df['periodeDebut'].dt.month_name()
        
        # Ajout de la semaine du mois
        df['semaine_mois'] = df['periodeDebut'].apply(get_week_of_month)
        
        # Pour une analyse plus précise, nous pouvons également considérer la période de fin
        df['annee_fin'] = df['periodeFin'].dt.year
        df['trimestre_fin'] = df['periodeFin'].apply(get_quarter)
        df['semestre_fin'] = df['periodeFin'].apply(get_semester)
        
        # Calcul de la durée de facturation en jours
        df['duree_facturation'] = (df['periodeFin'] - df['periodeDebut']).dt.days
        
        # Calcul de la consommation moyenne par jour
        df['conso_par_jour'] = df['Conso'] / df['duree_facturation'].replace(0, 1)  # Éviter la division par zéro
        
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return pd.DataFrame()
    
def prediction_dashboard(df_filtered, selected_year, selected_city, selected_client_type, selected_payment_status, period_type, period_label):
    st.title("Prediction Models")












import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import calendar
from datetime import datetime, timedelta
from dbConnection import get_connection  # Utilise votre fonction de connexion existante
import statsmodels.api as sm
from scipy import stats
import time


# Fonction pour obtenir le trimestre à partir d'une date
def get_quarter(date):
    """Returns the quarter (1-4) for a given date"""
    return (pd.to_datetime(date).month - 1) // 3 + 1

# Fonction pour obtenir le semestre à partir d'une date
def get_semester(date):
    """Returns the semester (1-2) for a given date"""
    return 1 if pd.to_datetime(date).month <= 6 else 2

# Fonction pour convertir un numéro de trimestre en étiquette
def get_quarter_label(quarter):
    """Converts a quarter number to label"""
    return f"Q{quarter}"

# Fonction pour convertir un numéro de semestre en étiquette
def get_semester_label(semester):
    """Converts a semester number to label"""
    return f"S{semester}"

# Fonction pour calculer la semaine du mois
def get_week_of_month(date):
    """Calculates the week of the month (1-5) for a given date"""
    date = pd.to_datetime(date)
    first_day = date.replace(day=1)
    dom = date.day
    adjusted_dom = dom + first_day.weekday()
    return (adjusted_dom - 1) // 7 + 1

# Fonction pour le chargement et la préparation des données
@st.cache_data
def load_prediction_data():
    try:
        engine = get_connection()
        
        # Exécuter une requête SQL pour récupérer toutes les données de la table consumption
        query = "SELECT * FROM consumption"
        df = pd.read_sql(query, engine)
        
        # Conversion des colonnes de date
        df['periodeDebut'] = pd.to_datetime(df['periodeDebut'])
        df['periodeFin'] = pd.to_datetime(df['periodeFin'])
        
        # Ajout des colonnes d'analyse temporelle
        df['annee_debut'] = df['periodeDebut'].dt.year
        df['trimestre_debut'] = df['periodeDebut'].apply(get_quarter)
        df['semestre_debut'] = df['periodeDebut'].apply(get_semester)
        df['mois_debut'] = df['periodeDebut'].dt.month
        df['nom_mois_debut'] = df['periodeDebut'].dt.month_name()
        
        # Ajout de la semaine du mois
        df['semaine_mois'] = df['periodeDebut'].apply(get_week_of_month)
        
        # Pour une analyse plus précise, nous pouvons également considérer la période de fin
        df['annee_fin'] = df['periodeFin'].dt.year
        df['trimestre_fin'] = df['periodeFin'].apply(get_quarter)
        df['semestre_fin'] = df['periodeFin'].apply(get_semester)
        
        # Calcul de la durée de facturation en jours
        df['duree_facturation'] = (df['periodeFin'] - df['periodeDebut']).dt.days
        
        # Calcul de la consommation moyenne par jour
        df['conso_par_jour'] = df['Conso'] / df['duree_facturation'].replace(0, 1)  # Éviter la division par zéro
        
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return pd.DataFrame()



def plot_residuals(y_true, y_pred, title="Analyse des résidus"):
    """
    Crée et affiche des graphiques d'analyse des résidus pour un modèle de régression.
    
    Args:
        y_true: Valeurs réelles (observées)
        y_pred: Valeurs prédites par le modèle
        title: Titre du graphique
    """
    residuals = y_true - y_pred
    
    # Créer une figure avec 2 sous-graphiques (2 colonnes, 1 ligne)
    fig = make_subplots(
        rows=1, 
        cols=2,
        subplot_titles=(
            "Résidus vs Valeurs prédites",
            "QQ-Plot (Normalité)"
        )
    )
    
    # 1. Résidus vs Valeurs prédites (colonne 1)
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            name="Résidus",
            marker=dict(color='blue', size=8)
        ),
        row=1, col=1
    )
    
    # Ajouter une ligne horizontale à y=0
    fig.add_trace(
        go.Scatter(
            x=[min(y_pred), max(y_pred)],
            y=[0, 0],
            mode='lines',
            name='Résidu zéro',
            line=dict(color='red', dash='dash')
        ),
        row=1, col=1
    )
    
    # 2. QQ-Plot pour la normalité (colonne 2)
    # Trier les résidus
    sorted_residuals = np.sort(residuals)
    
    # Calculer les quantiles théoriques de la distribution normale
    n = len(sorted_residuals)
    quantiles_theory = np.array([(i - 0.5) / n for i in range(1, n + 1)])
    quantiles_theory = stats.norm.ppf(quantiles_theory)
    
    fig.add_trace(
        go.Scatter(
            x=quantiles_theory,
            y=sorted_residuals,
            mode='markers',
            name='QQ-Plot',
            marker=dict(color='blue', size=8)
        ),
        row=1, col=2
    )
    
    # Ligne de référence pour la normalité parfaite
    min_val = min(min(quantiles_theory), min(sorted_residuals))
    max_val = max(max(quantiles_theory), max(sorted_residuals))
    
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Normalité parfaite',
            line=dict(color='red', dash='dash')
        ),
        row=1, col=2
    )
    
    # Mettre à jour la mise en page
    fig.update_layout(
        title=title,
        height=500,  # Réduit la hauteur puisque nous avons seulement 1 ligne maintenant
        width=1000,
        showlegend=False,
        hovermode='closest'
    )
    
    # Mettre à jour les axes
    fig.update_xaxes(title_text="Valeurs prédites", row=1, col=1)
    fig.update_xaxes(title_text="Quantiles théoriques", row=1, col=2)
    
    fig.update_yaxes(title_text="Résidus", row=1, col=1)
    fig.update_yaxes(title_text="Résidus", row=1, col=2)
    
    # Afficher le graphique
    st.plotly_chart(fig)
    
    # Ajouter une interprétation des graphiques
    st.write("### 📝 Interprétation des graphiques de résidus")
    
    st.write("""
    **1. Résidus vs Valeurs prédites (gauche):**
    - Les points doivent être répartis aléatoirement autour de la ligne horizontale zéro.
    - Aucune tendance ou forme particulière ne devrait être visible.
    - Des formes spécifiques (entonnoir, courbe) indiqueraient des violations des hypothèses.
    
    **2. QQ-Plot (droite):**
    - Si les résidus suivent une distribution normale, les points s'alignent sur la ligne diagonale rouge.
    - Des écarts significatifs par rapport à cette ligne indiquent une non-normalité.
    """)
    
    # Ajouter une évaluation globale des résidus
    st.write("### 🔍 Évaluation globale de la qualité du modèle")
    
    # Analyser l'autocorrélation des résidus (test de Durbin-Watson)
    try:
        dw_statistic = sm.stats.stattools.durbin_watson(residuals)
        if dw_statistic < 1.5:
            dw_interpretation = "**Autocorrélation positive détectée** : Les résidus ne sont pas indépendants, ce qui pourrait indiquer que le modèle manque de variables explicatives importantes."
        elif dw_statistic > 2.5:
            dw_interpretation = "**Autocorrélation négative détectée** : Les résidus oscillent de manière systématique, ce qui pourrait indiquer une surajustement ou des problèmes dans la spécification du modèle."
        else:
            dw_interpretation = "**Pas d'autocorrélation significative** : Les résidus semblent être indépendants, ce qui est positif pour la qualité du modèle."
        
        st.write(f"**Test de Durbin-Watson:** {dw_statistic:.3f} - {dw_interpretation}")
    except:
        st.write("Impossible de calculer la statistique de Durbin-Watson")
    
    # Vérifier la normalité (test de Shapiro-Wilk)
    try:
        if len(residuals) <= 5000:  # Le test de Shapiro-Wilk est limité à environ 5000 observations
            shapiro_test = stats.shapiro(residuals)
            shapiro_p_value = shapiro_test[1]
            
            if shapiro_p_value < 0.05:
                shapiro_interpretation = "**Non-normalité détectée** : Les résidus ne suivent pas une distribution normale, ce qui peut affecter la fiabilité des intervalles de confiance et des tests d'hypothèse."
            else:
                shapiro_interpretation = "**Normalité confirmée** : Les résidus semblent suivre une distribution normale, ce qui est positif pour la qualité du modèle."
            
            st.write(f"**Test de Shapiro-Wilk (normalité):** p-value = {shapiro_p_value:.4f} - {shapiro_interpretation}")
    except:
        st.write("Impossible de calculer le test de normalité de Shapiro-Wilk")
    
    # Conseil général basé sur l'analyse visuelle
    std_norm_residuals = residuals / np.std(residuals)
    outliers = sum(abs(std_norm_residuals) > 2)
    outliers_percent = (outliers / len(residuals)) * 100
    
    if outliers_percent > 10:
        st.write(f"⚠️ **{outliers_percent:.1f}%** des résidus sont des valeurs aberrantes (> 2 écarts-types). Cela suggère que le modèle pourrait ne pas être adapté à certaines observations.")
    else:
        st.write(f"✅ Seulement **{outliers_percent:.1f}%** des résidus sont des valeurs aberrantes (> 2 écarts-types), ce qui est acceptable.")


# Fonctions pour les métriques d'évaluation du modèle
def evaluate_model(y_true, y_pred):
    """Calcule et retourne les métriques d'évaluation pour un modèle"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calcul du MAPE (Mean Absolute Percentage Error) en évitant la division par zéro
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else np.nan
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE (%)': mape
    }

# Fonction pour afficher les métriques d'évaluation
def display_metrics(metrics):
    # Créer un dataframe pour afficher les métriques
    metrics_df = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
    
    # Formater les valeurs
    metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
    
    # Afficher le tableau des métriques
    st.write("### Métriques d'évaluation du modèle")
    st.table(metrics_df.set_index('Metric'))
    
    # Interprétation des métriques
    r2 = float(metrics['R²'])
    mape = float(metrics['MAPE (%)']) if not pd.isna(metrics['MAPE (%)']) else float('inf')
    
    if r2 >= 0.8:
        r2_interpretation = "Le modèle explique **très bien** la variance des données (R² élevé)."
    elif r2 >= 0.6:
        r2_interpretation = "Le modèle explique **assez bien** la variance des données (R² modéré)."
    elif r2 >= 0.4:
        r2_interpretation = "Le modèle explique **moyennement** la variance des données (R² moyen)."
    else:
        r2_interpretation = "Le modèle explique **peu** la variance des données (R² faible)."
    
    if mape < 10:
        mape_interpretation = "Le modèle a une **très bonne** précision avec une erreur moyenne de pourcentage absolue faible."
    elif mape < 20:
        mape_interpretation = "Le modèle a une **bonne** précision avec une erreur moyenne de pourcentage absolue acceptable."
    elif mape < 30:
        mape_interpretation = "Le modèle a une précision **moyenne** avec une erreur moyenne de pourcentage absolue modérée."
    else:
        mape_interpretation = "Le modèle a une **faible** précision avec une erreur moyenne de pourcentage absolue élevée."
    
    st.write("### 🔍 Interprétation des métriques")
    st.write(r2_interpretation)
    
    if not pd.isna(metrics['MAPE (%)']):
        st.write(mape_interpretation)
    else:
        st.write("Le MAPE n'a pas pu être calculé (division par zéro possible).")

# Fonction pour tracer la comparaison entre les valeurs réelles et prédites
def plot_predicted_vs_actual(X, y_true, y_pred, title, x_label):
    # Créer un dataframe pour le tracé
    plot_df = pd.DataFrame({
        'X': X,
        'Actual': y_true,
        'Predicted': y_pred
    })
    
    # Créer un graphique interactif avec Plotly
    fig = make_subplots()
    
    # Ajouter les valeurs réelles
    fig.add_trace(
        go.Scatter(
            x=plot_df['X'],
            y=plot_df['Actual'],
            mode='markers+lines',
            name='Valeurs réelles',
            marker=dict(size=10, color='blue')
        )
    )
    
    # Ajouter les valeurs prédites
    fig.add_trace(
        go.Scatter(
            x=plot_df['X'],
            y=plot_df['Predicted'],
            mode='markers+lines',
            name='Valeurs prédites',
            marker=dict(size=10, color='red')
        )
    )
    
    # Mettre à jour la mise en page
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title='Consommation',
        legend=dict(x=0, y=1, traceorder='normal'),
        height=500
    )
    
    # Afficher le graphique
    st.plotly_chart(fig)

def create_sidebar_filters(df):
    st.sidebar.markdown("## Filtres")
    
    # Filtre utilisateur - liste tous les identifiants
    all_users = ["All users"] + sorted(df['identifier'].unique())
    selected_user = st.sidebar.selectbox("Utilisateur", all_users)
    
    # Autres filtres
    selected_year = st.sidebar.selectbox("Année", sorted(df['annee_debut'].unique(), reverse=True))
    # selected_city = st.sidebar.selectbox("Ville", ["All cities"] + sorted(df['Ville'].unique()))
    # selected_client_type = st.sidebar.selectbox("Type de Client", ["All types"] + sorted(df['TypeClient'].unique()))
    # selected_payment_status = st.sidebar.selectbox("Statut de Paiement", ["All statuses"] + sorted(df['StatutPaiement'].unique()))
    
    #Options de prédiction basées sur la période
    st.sidebar.markdown("## Options de Prédiction")
    prediction_timeframe = st.sidebar.radio(
        "Prédire pour:",
         ["Mois prochain", "Trimestre prochain", "Semestre prochain", "Année prochaine"]
     )
    
    return selected_user, selected_year, #selected_city, selected_client_type, selected_payment_status, prediction_timeframe

# Fonction pour appliquer les filtres
def apply_filters(df, selected_user, selected_year, selected_city, selected_client_type, selected_payment_status):
    """Applique tous les filtres au dataframe"""
    # Faire une copie pour éviter de modifier l'original
    filtered_df = df.copy()
    
    # Appliquer le filtre utilisateur
    if selected_user != "All users":
        filtered_df = filtered_df[filtered_df['ID_Utilisateur'] == selected_user]
    
    # Appliquer les autres filtres
    if selected_year is not None:
        filtered_df = filtered_df[filtered_df['annee_debut'] == selected_year]
    
    if selected_city != 'All cities':
        filtered_df = filtered_df[filtered_df['Ville'] == selected_city]
    
    if selected_client_type != 'All types':
        filtered_df = filtered_df[filtered_df['TypeClient'] == selected_client_type]
    
    if selected_payment_status != 'All statuses':
        filtered_df = filtered_df[filtered_df['StatutPaiement'] == selected_payment_status]
    
    return filtered_df

# Fonction modifiée pour la prédiction par période
# Fonction modifiée pour la prédiction par période avec support pour peu de données
# Fonction modifiée pour la prédiction par période avec support pour peu de données
# Fonction modifiée pour la prédiction par période avec support pour peu de données
def predict_by_period(df, period_type, prediction_timeframe, selected_year):
    # Adapter le titre en fonction du type de prédiction
    st.write(f"## 📈 Prédiction de consommation pour {prediction_timeframe}")
    
    # Déterminer la colonne de période en fonction du type de prédiction
    if "mois" in prediction_timeframe.lower():
        period_column = 'mois_debut'
        x_label = 'Mois'
        time_unit = 'mois'
    elif "trimestre" in prediction_timeframe.lower():
        period_column = 'trimestre_debut'
        x_label = 'Trimestre'
        time_unit = 'trimestre'
    elif "semestre" in prediction_timeframe.lower():
        period_column = 'semestre_debut'
        x_label = 'Semestre'
        time_unit = 'semestre'
    elif "année" in prediction_timeframe.lower() or "annee" in prediction_timeframe.lower():
        period_column = 'annee_debut'
        x_label = 'Année'
        time_unit = 'année'
    else:
        # Par défaut, utiliser le mois si le format n'est pas reconnu
        st.info(f"Format de période '{prediction_timeframe}' non reconnu, utilisation du mois par défaut")
        period_column = 'mois_debut'
        x_label = 'Mois'
        time_unit = 'mois'

    # Ajouter ce code pour voir s'il y a une tendance temporelle
    if time_unit == 'mois':
        agg_data = df.groupby('mois_debut')['Conso'].mean().reset_index()
    elif time_unit == 'trimestre':
        agg_data = df.groupby('trimestre_debut')['Conso'].mean().reset_index()
    elif time_unit == 'semestre':
        agg_data = df.groupby('semestre_debut')['Conso'].mean().reset_index()
    else:
        agg_data = df.groupby('annee_debut')['Conso'].mean().reset_index()
    
    st.write("Variation de consommation par période:")
    st.table(agg_data)
    
    # Utiliser l'année sélectionnée au lieu de l'année actuelle
    current_year = selected_year
    
    # Déterminer la période actuelle et la période suivante à prédire
    if time_unit == 'mois':
        available_months = sorted(df[df['annee_debut'] == current_year]['mois_debut'].unique())
        if not available_months:
            available_months = sorted(df['mois_debut'].unique())
            if not available_months:
                st.warning(f"⚠️ Pas de données pour le mois")
                return None, None, None, None
            
            current_period = available_months[-1]
            st.info(f"Utilisation des données de toutes les années disponibles pour le mois {current_period}")
        else:
            current_period = available_months[-1]
            
        next_period = current_period + 1 if current_period < 12 else 1
        next_period_year = current_year if next_period > current_period else current_year + 1
        current_period_name = calendar.month_name[current_period]
        next_period_name = calendar.month_name[next_period]
    
    elif time_unit == 'trimestre':
        available_quarters = sorted(df[df['annee_debut'] == current_year]['trimestre_debut'].unique())
        if not available_quarters:
            available_quarters = sorted(df['trimestre_debut'].unique())
            if not available_quarters:
                st.warning(f"⚠️ Pas de données pour le trimestre")
                return None, None, None, None
                
            current_period = available_quarters[-1]
            st.info(f"Utilisation des données de toutes les années disponibles pour le trimestre {current_period}")
        else:
            current_period = available_quarters[-1]
            
        next_period = current_period + 1 if current_period < 4 else 1
        next_period_year = current_year if next_period > current_period else current_year + 1
        current_period_name = f"Q{current_period}"
        next_period_name = f"Q{next_period}"
    
    elif time_unit == 'semestre':
        available_semesters = sorted(df[df['annee_debut'] == current_year]['semestre_debut'].unique())
        if not available_semesters:
            available_semesters = sorted(df['semestre_debut'].unique())
            if not available_semesters:
                st.warning(f"⚠️ Pas de données pour le semestre")
                return None, None, None, None
                
            current_period = available_semesters[-1]
            st.info(f"Utilisation des données de toutes les années disponibles pour le semestre {current_period}")
        else:
            current_period = available_semesters[-1]
            
        next_period = 2 if current_period == 1 else 1
        next_period_year = current_year if next_period > current_period else current_year + 1
        current_period_name = f"S{current_period}"
        next_period_name = f"S{next_period}"
    
    else:  # année
        current_period = current_year
        next_period = current_period + 1
        next_period_year = next_period
        current_period_name = str(current_period)
        next_period_name = str(next_period)
    
    # Filtrer les données pour la période d'entraînement
    if time_unit == 'année':
        df_train = df[df[period_column] < current_period]
        if len(df_train) < 3:
            df_train = df.copy()
            st.info("Utilisation de toutes les années disponibles pour l'entraînement")
    else:
        if len(df[(df[period_column] == current_period) & (df['annee_debut'] == current_year)]) < 3:
            df_train = df[df[period_column] == current_period].copy()
            st.info(f"Utilisation de toutes les données historiques pour {time_unit} {current_period_name}")
            
            if len(df_train) < 3:
                df_train = df.copy()
                st.info(f"Utilisation de toutes les données disponibles pour l'entraînement du modèle")
        else:
            df_train = df[(df[period_column] == current_period) & (df['annee_debut'] == current_year)].copy()
    
    # Réinitialiser les indices
    df_train = df_train.reset_index(drop=True)
    
    # Vérification de la quantité de données
    if len(df_train) < 2:
        st.warning(f"⚠️ Pas assez de données disponibles pour entraîner un modèle.")
        
        if len(df_train) > 0:
            simple_pred = df_train['Conso'].mean()
            st.info(f"Utilisation d'une moyenne simple pour la prédiction: {simple_pred:.2f}")

            st.write(f"### Prédiction pour {time_unit} {next_period_name} ({next_period_year})")
            st.write(f"Basé sur les données limitées disponibles, la consommation prévue pour {time_unit} {next_period_name} ({next_period_year}) est:")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label=f"Prédiction pour {next_period_name} ({next_period_year})",
                    value=f"{simple_pred:.2f}",
                    delta="N/A"
                )
            with col2:
                st.metric(
                    label="Estimation basse",
                    value=f"{max(0, simple_pred * 0.8):.2f}"
                )
            with col3:
                st.metric(
                    label="Estimation haute",
                    value=f"{simple_pred * 1.2:.2f}"
                )
            
            return None, current_period, next_period, simple_pred
        
        return None, None, None, None
    
    # Préparation des caractéristiques
    # IMPORTANT: Pour la prédiction par période, nous utilisons explicitement le numéro de période
    # comme caractéristique, pas seulement comme filtre
    available_features = []
    
    if len(df_train) < 5:
        # Utiliser uniquement la période comme caractéristique
        X_train = df_train[period_column].values.reshape(-1, 1)
        available_features = [period_column]
    else:
        try:
            # Liste des caractéristiques potentielles
            potential_features = [period_column, 'duree_facturation']
            
            # Filtrer les caractéristiques effectivement disponibles
            available_features = [f for f in potential_features if f in df_train.columns]
            
            X_train = df_train[available_features].values
        except:
            # En cas d'erreur, revenir à la caractéristique unique
            X_train = df_train[period_column].values.reshape(-1, 1)
            available_features = [period_column]
    
    y_train = df_train['Conso'].values
    
    # CRÉER LE MODÈLE
    model = LinearRegression()
    
    try:
        model.fit(X_train, y_train)

        if model is not None:
            st.write("### Test direct du modèle")
            test_values = [current_period, next_period, current_period+2]
            for val in test_values:
                pred = model.predict(np.array([[val]]))[0]
                st.write(f"Pour période {val}: prédiction = {pred:.2f}")
            
            st.write(f"Coefficients du modèle: {model.coef_}")
            st.write(f"Intercept du modèle: {model.intercept_}")
        
        # Faire des prédictions pour la période actuelle
        y_pred_current = model.predict(X_train)
        
        # Évaluer le modèle
        current_metrics = evaluate_model(y_train, y_pred_current)

        # Afficher les métriques pour la période actuelle
        st.write(f"## 📊 Performance sur les données d'entraînement")
        display_metrics(current_metrics)
        
        # Ajouter l'analyse des résidus
        st.write("## 📈 Analyse des résidus du modèle")
        try:
            plot_residuals(y_train, y_pred_current, title=f"Analyse des résidus pour la prédiction par {time_unit}")
        except Exception as e:
            st.warning(f"Impossible d'afficher l'analyse des résidus: {str(e)}")

        # Préparation de X_forecast pour la prédiction
        if len(available_features) == 1:
            # Une seule caractéristique: la période
            X_forecast = np.array([[next_period]])
        else:
            # Plusieurs caractéristiques
            # IMPORTANT: S'assurer que X_forecast a le même format que X_train
            features_dict = {}
            for i, feature in enumerate(available_features):
                if feature == period_column:
                    features_dict[feature] = next_period
                elif feature == 'duree_facturation':
                    features_dict[feature] = df_train['duree_facturation'].mean()
            
            X_forecast = np.array([[features_dict[feature] for feature in available_features]])
        
        # Débogage
        st.write(f"Debug - X_train shape: {X_train.shape}")
        st.write(f"Debug - X_forecast shape: {X_forecast.shape}")
        if len(X_train) > 0:
            st.write(f"Debug - X_train exemple: {X_train[0]}")
        st.write(f"Debug - X_forecast: {X_forecast[0]}")
            
        # Prédictions pour comparaison
        if len(available_features) == 1:
            # Test avec différentes valeurs de période
            test_pred_current = model.predict(np.array([[current_period]]))[0]
            test_pred_next = model.predict(np.array([[next_period]]))[0]
            st.write(f"Debug - Test période {current_period}: {test_pred_current:.2f}")
            st.write(f"Debug - Test période {next_period}: {test_pred_next:.2f}")
        
        # Faire la prédiction pour la période suivante
        next_period_prediction = model.predict(X_forecast)[0]
        
    except Exception as e:
        st.warning(f"⚠️ Erreur lors de l'entraînement du modèle: {str(e)}")
        st.info("Utilisation d'une moyenne simple pour la prédiction")
        
        # Faire une prédiction simple (moyenne)
        next_period_prediction = df_train['Conso'].mean()
        current_metrics = {
            'MSE': np.mean((df_train['Conso'] - next_period_prediction) ** 2),
            'RMSE': np.sqrt(np.mean((df_train['Conso'] - next_period_prediction) ** 2)),
            'MAE': np.mean(np.abs(df_train['Conso'] - next_period_prediction)),
            'R²': 0,
            'MAPE (%)': np.nan
        }
    
    # Afficher la prédiction
    st.write(f"### Prédiction pour {time_unit} {next_period_name} ({next_period_year})")
    st.write(f"Basé sur les données disponibles, la consommation prévue pour {time_unit} {next_period_name} ({next_period_year}) est:")
    
    # Afficher la prédiction avec intervalle de confiance
    confidence = 0.95  # 95% de confiance
    mse = current_metrics['MSE']
    confidence_interval = 1.96 * np.sqrt(mse)  # Intervalle de confiance approximatif à 95%
    
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_current = df_train['Conso'].mean()
        delta = next_period_prediction - avg_current
        st.metric(
            label=f"Prédiction pour {next_period_name} ({next_period_year})",
            value=f"{next_period_prediction:.2f}",
            delta=f"{delta:.2f}" if not np.isnan(delta) else "N/A"
        )
    with col2:
        st.metric(
            label="Borne inférieure",
            value=f"{max(0, next_period_prediction - confidence_interval):.2f}"
        )
    with col3:
        st.metric(
            label="Borne supérieure",
            value=f"{next_period_prediction + confidence_interval:.2f}"
        )
    
    # Afficher l'historique et la comparaison avec la prédiction
    if len(df_train) >= 2:
        st.write(f"### 📈 Comparaison historique et prédiction")
        
        # Préparer les données pour le tableau et le graphique
        period_predictions = []
        
        # Déterminer le libellé de la période actuelle et future
        if time_unit == 'mois':
            current_label = calendar.month_name[current_period]
            next_label = calendar.month_name[next_period]
        elif time_unit == 'trimestre':
            current_label = f"Q{current_period}"
            next_label = f"Q{next_period}"
        elif time_unit == 'semestre':
            current_label = f"S{current_period}"
            next_label = f"S{next_period}"
        else:  # année
            current_label = str(current_year)
            next_label = str(next_period_year)
        
        # Calculer la consommation moyenne de la période actuelle
        current_conso = df_train['Conso'].mean()
        
        # Calculer la variation en pourcentage
        variation_pct = ((next_period_prediction - current_conso) / current_conso * 100) if current_conso > 0 else 0
        
        # Préparer les données comme pour les graphiques par ville/type client
        period_predictions.append({
            'Période': current_label,
            f'Consommation {current_label} ({current_year})': current_conso,
            f'Prédiction {next_label} ({next_period_year})': next_period_prediction,
            'Variation (%)': variation_pct
        })
        
        # Créer un DataFrame avec les prédictions
        period_pred_df = pd.DataFrame(period_predictions)
        
        # Afficher le tableau des prédictions
        st.table(period_pred_df.style.format({
            f'Consommation {current_label} ({current_year})': '{:.2f}',
            f'Prédiction {next_label} ({next_period_year})': '{:.2f}',
            'Variation (%)': '{:.2f}'
        }))
        
        # Créer un graphique à barres
        fig = go.Figure()
        
        # Ajouter les valeurs actuelles
        fig.add_trace(
            go.Bar(
                x=[f"{current_label} ({current_year})"],
                y=[current_conso],
                name=f'Consommation actuelle',
                marker_color='blue'
            )
        )
        
        # Ajouter les valeurs prédites
        fig.add_trace(
            go.Bar(
                x=[f"{next_label} ({next_period_year})"],
                y=[next_period_prediction],
                name=f'Prédiction',
                marker_color='red'
            )
        )
        
        # Mettre à jour la mise en page
        fig.update_layout(
            title=f"Comparaison de la consommation actuelle et prédite",
            xaxis_title="Période",
            yaxis_title="Consommation",
            barmode='group',
            height=500
        )
        
        # Afficher le graphique à barres
        st.plotly_chart(fig)
        
        # Afficher le graphique de tendance historique
        if len(df[period_column].unique()) >= 3:
            st.write(f"### 📊 Tendance historique et projection")
            
            # Données historiques selon le type de période
            if time_unit == 'mois':
                periods = []
                values = []
                
                for year in sorted(df['annee_debut'].unique()):
                    month_data = df[(df['annee_debut'] == year) & (df['mois_debut'] == current_period)]
                    if not month_data.empty:
                        periods.append(f"{calendar.month_abbr[current_period]}-{year}")
                        values.append(month_data['Conso'].mean())
                
                periods.append(f"{calendar.month_abbr[next_period]}-{next_period_year}")
                values.append(next_period_prediction)
                
            elif time_unit == 'trimestre':
                periods = []
                values = []
                
                for year in sorted(df['annee_debut'].unique()):
                    quarter_data = df[(df['annee_debut'] == year) & (df['trimestre_debut'] == current_period)]
                    if not quarter_data.empty:
                        periods.append(f"Q{current_period}-{year}")
                        values.append(quarter_data['Conso'].mean())
                
                periods.append(f"Q{next_period}-{next_period_year}")
                values.append(next_period_prediction)
                
            elif time_unit == 'semestre':
                periods = []
                values = []
                
                for year in sorted(df['annee_debut'].unique()):
                    semester_data = df[(df['annee_debut'] == year) & (df['semestre_debut'] == current_period)]
                    if not semester_data.empty:
                        periods.append(f"S{current_period}-{year}")
                        values.append(semester_data['Conso'].mean())
                
                periods.append(f"S{next_period}-{next_period_year}")
                values.append(next_period_prediction)
                
            else:  # année
                periods = [str(y) for y in sorted(df['annee_debut'].unique())]
                values = [df[df['annee_debut'] == y]['Conso'].mean() for y in sorted(df['annee_debut'].unique())]
                
                periods.append(str(next_period))
                values.append(next_period_prediction)
            
            if len(periods) >= 2:
                fig = go.Figure()
                
                if len(periods) > 1:
                    fig.add_trace(
                        go.Scatter(
                            x=periods[:-1],
                            y=values[:-1],
                            mode='lines+markers',
                            name='Données historiques',
                            line=dict(color='blue')
                        )
                    )
                
                fig.add_trace(
                    go.Scatter(
                        x=[periods[-1]],
                        y=[values[-1]],
                        mode='markers',
                        name='Prédiction',
                        marker=dict(color='red', size=12, symbol='star')
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=[periods[-1], periods[-1]],
                        y=[max(0, values[-1] - confidence_interval), values[-1] + confidence_interval],
                        mode='lines',
                        name='Intervalle de confiance (95%)',
                        line=dict(color='red', width=1, dash='dash')
                    )
                )
                
                fig.update_layout(
                    title=f"Tendance historique de la consommation et projection",
                    xaxis_title="Période",
                    yaxis_title="Consommation",
                    legend=dict(x=0, y=1, traceorder='normal'),
                    height=500
                )
                
                st.plotly_chart(fig)
    
    # Analyse des résultats
    st.write("### 📝 Interprétation des résultats")
    
    # Calculer la variation en pourcentage
    avg_current = df_train['Conso'].mean()
    if avg_current > 0:
        percent_change = ((next_period_prediction - avg_current) / avg_current) * 100
        
        # Tendance globale
        st.write("#### Tendance globale")
        if percent_change > 5:
            st.write(f"🔺 La prédiction montre une **augmentation** de {percent_change:.1f}% pour {time_unit} {next_period_name} par rapport aux données disponibles.")
        elif percent_change > 0:
            st.write(f"↗️ La prédiction montre une **légère augmentation** de {percent_change:.1f}% pour {time_unit} {next_period_name} par rapport aux données disponibles.")
        elif percent_change > -5:
            st.write(f"↘️ La prédiction montre une **légère baisse** de {abs(percent_change):.1f}% pour {time_unit} {next_period_name} par rapport aux données disponibles.")
        else:
            st.write(f"🔻 La prédiction montre une **baisse** de {abs(percent_change):.1f}% pour {time_unit} {next_period_name} par rapport aux données disponibles.")
    else:
        st.write("La variation en pourcentage ne peut pas être calculée car la consommation moyenne actuelle est nulle ou négative.")
    
    # Fiabilité de la prédiction
    st.write("#### Fiabilité de la prédiction")
    
    if len(df_train) < 5:
        st.write("⚠️ Prédiction basée sur **très peu de données** (<5 points). Les résultats doivent être interprétés avec prudence.")
        st.write("💡 Pour améliorer la fiabilité des prédictions, essayez d'élargir vos filtres ou d'obtenir plus de données historiques.")
    else:
        r2 = current_metrics['R²']
        if r2 > 0.8:
            st.write("🎯 La prédiction est considérée comme **assez fiable** (R² élevé).")
        elif r2 > 0.6:
            st.write("✅ La prédiction est considérée comme **modérément fiable** (R² acceptable).")
        elif r2 > 0.4:
            st.write("⚠️ La prédiction est considérée comme **peu fiable** (R² moyen), les tendances historiques présentant une certaine variabilité.")
        else:
            st.write("⚠️ La prédiction est considérée comme **très peu fiable** (R² faible), les tendances historiques étant très variables ou les données insuffisantes.")
    
    return model, current_period, next_period, next_period_prediction
def predict_by_city(df, prediction_timeframe, selected_year):
    st.write(f"## 📈 Prédiction de consommation par ville pour {prediction_timeframe}")
    
    # Déterminer la période en fonction du type de prédiction
    if "mois" in prediction_timeframe.lower():
        period_column = 'mois_debut'
        x_label = 'Mois'
        time_unit = 'mois'
    elif "trimestre" in prediction_timeframe.lower():
        period_column = 'trimestre_debut'
        x_label = 'Trimestre'
        time_unit = 'trimestre'
    elif "semestre" in prediction_timeframe.lower():
        period_column = 'semestre_debut'
        x_label = 'Semestre'
        time_unit = 'semestre'
    elif "année" in prediction_timeframe.lower() or "annee" in prediction_timeframe.lower():
        period_column = 'annee_debut'
        x_label = 'Année'
        time_unit = 'année'
    else:
        # Par défaut, utiliser le mois si le format n'est pas reconnu
        st.info(f"Format de période '{prediction_timeframe}' non reconnu, utilisation du mois par défaut")
        period_column = 'mois_debut'
        x_label = 'Mois'
        time_unit = 'mois'
    
    # Utiliser l'année sélectionnée
    current_year = selected_year
    
    # Déterminer la période actuelle (selon l'année sélectionnée) et la période suivante à prédire
    if time_unit == 'mois':
        available_months = sorted(df[df['annee_debut'] == current_year]['mois_debut'].unique())
        if not available_months:
            # MODIFICATION: Si pas de données pour l'année sélectionnée, utiliser toutes les années
            available_months = sorted(df['mois_debut'].unique())
            if not available_months:
                st.warning(f"⚠️ Pas de données pour le mois")
                return None, None, None
                
            # Utiliser le dernier mois disponible dans toutes les données
            current_period = available_months[-1]
            st.info(f"Utilisation des données de toutes les années disponibles pour le mois {current_period}")
        else:
            current_period = available_months[-1]  # Dernier mois disponible
            
        next_period = current_period + 1 if current_period < 12 else 1
        next_period_year = current_year if next_period > current_period else current_year + 1
        current_period_name = calendar.month_name[current_period]
        next_period_name = calendar.month_name[next_period]
    
    elif time_unit == 'trimestre':
        available_quarters = sorted(df[df['annee_debut'] == current_year]['trimestre_debut'].unique())
        if not available_quarters:
            # MODIFICATION: Si pas de données pour l'année sélectionnée, utiliser toutes les années
            available_quarters = sorted(df['trimestre_debut'].unique())
            if not available_quarters:
                st.warning(f"⚠️ Pas de données pour le trimestre")
                return None, None, None
                
            # Utiliser le dernier trimestre disponible dans toutes les données
            current_period = available_quarters[-1]
            st.info(f"Utilisation des données de toutes les années disponibles pour le trimestre {current_period}")
        else:
            current_period = available_quarters[-1]  # Dernier trimestre disponible
            
        next_period = current_period + 1 if current_period < 4 else 1
        next_period_year = current_year if next_period > current_period else current_year + 1
        current_period_name = f"Q{current_period}"
        next_period_name = f"Q{next_period}"
    
    elif time_unit == 'semestre':
        available_semesters = sorted(df[df['annee_debut'] == current_year]['semestre_debut'].unique())
        if not available_semesters:
            # MODIFICATION: Si pas de données pour l'année sélectionnée, utiliser toutes les années
            available_semesters = sorted(df['semestre_debut'].unique())
            if not available_semesters:
                st.warning(f"⚠️ Pas de données pour le semestre")
                return None, None, None
                
            # Utiliser le dernier semestre disponible dans toutes les données
            current_period = available_semesters[-1]
            st.info(f"Utilisation des données de toutes les années disponibles pour le semestre {current_period}")
        else:
            current_period = available_semesters[-1]  # Dernier semestre disponible
            
        next_period = 2 if current_period == 1 else 1
        next_period_year = current_year if next_period > current_period else current_year + 1
        current_period_name = f"S{current_period}"
        next_period_name = f"S{next_period}"
    
    else:  # année
        current_period = current_year
        next_period = current_period + 1
        next_period_year = next_period
        current_period_name = str(current_period)
        next_period_name = str(next_period)
    
    # Filtrer les données pour la période actuelle
    df_current = df[(df[period_column] == current_period) & (df['annee_debut'] == current_year)]
    
    # MODIFICATION: Si pas assez de données pour la période actuelle, essayer d'utiliser toutes les données de cette période
    if len(df_current) < 5 or len(df_current['Ville'].unique()) < 2:
        st.info(f"Pas assez de données pour {time_unit} {current_period_name} de l'année {current_year}. Utilisation de toutes les années disponibles.")
        df_current = df[df[period_column] == current_period]
        
        # Si toujours pas assez de données
        if len(df_current) < 5 or len(df_current['Ville'].unique()) < 2:
            st.warning(f"⚠️ Toujours pas assez de données pour {time_unit} {current_period_name}. Utilisation de toutes les données disponibles.")
            df_current = df.copy()
    
    # Vérifier s'il y a suffisamment de villes et de données après les tentatives d'élargissement
    if len(df_current['Ville'].unique()) < 2:
        st.warning("⚠️ Pas assez de villes différentes pour créer un modèle. Impossible de faire des prédictions par ville.")
        return None, None, None
    
    if len(df_current) < 2:
        st.warning(f"⚠️ Pas assez de données pour entraîner un modèle fiable.")
        return None, None, None
    
    # Réinitialiser les indices
    df_current = df_current.reset_index(drop=True)
    
    # Encoder les villes (one-hot encoding)
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    cities_encoded = encoder.fit_transform(df_current[['Ville']])
    
    # Créer un dataframe avec les villes encodées
    cities_df = pd.DataFrame(
        cities_encoded,
        columns=[f"ville_{city}" for city in encoder.categories_[0][1:]]
    )
    
    # Concaténer avec les autres caractéristiques
    try:
        X = pd.concat([df_current[[period_column]], cities_df], axis=1)
        y = df_current['Conso'].values
        
        # Créer et entraîner le modèle
        model = LinearRegression()
        model.fit(X, y)
        
        # Faire des prédictions pour la période actuelle (pour évaluer le modèle)
        y_pred = model.predict(X)
        
        # Évaluer le modèle
        metrics = evaluate_model(y, y_pred)
    except Exception as e:
        st.warning(f"⚠️ Erreur lors de l'entraînement du modèle: {str(e)}")
        st.info("Utilisation d'une approche simplifiée")
        # Approche simplifiée en cas d'erreur
        metrics = {
            'MSE': np.nan,
            'RMSE': np.nan,
            'MAE': np.nan,
            'R²': np.nan,
            'MAPE (%)': np.nan
        }
        # Continuer avec un modèle simpliste
        model = None
    
    # Afficher les métriques si un modèle a été créé
    if model is not None:
        st.write(f"## 📊 Performance sur les données de {time_unit} {current_period_name}")
        display_metrics(metrics)


    # if model is not None:
    #     st.write(f"## 📊 Performance sur les données de {time_unit} {current_period_name}")
    #     display_metrics(metrics)
        
        # NOUVEAU: Ajouter l'analyse des résidus
        st.write("## 📈 Analyse des résidus du modèle")
        
        try:
            # Utiliser la fonction plot_residuals pour afficher les graphiques des résidus
            plot_residuals(y, y_pred, title=f"Analyse des résidus pour la prédiction par ville")
        except Exception as e:
            st.warning(f"Impossible d'afficher l'analyse des résidus: {str(e)}")

    
    # Prédiction pour chaque ville pour la période suivante
    st.write(f"## 🏙️ Prédiction de consommation par ville pour {time_unit} {next_period_name}")
    
    # Obtenir la liste des villes uniques
    cities = df_current['Ville'].unique()
    
    # Préparer un dataframe pour les prédictions par ville
    city_predictions = []
    
    # Pour chaque ville, prédire la consommation
    for city in cities:
        try:
            if model is not None:
                # Créer un échantillon pour cette ville pour la période suivante
                next_period_sample = pd.DataFrame({
                    period_column: [next_period]  # Utiliser la période suivante
                })
                
                # Encoder la ville
                city_encoded = pd.DataFrame(columns=[f"ville_{c}" for c in encoder.categories_[0][1:]])
                for col in city_encoded.columns:
                    city_encoded[col] = [1 if col == f"ville_{city}" and city != encoder.categories_[0][0] else 0]
                
                # Concaténer les caractéristiques
                city_features = pd.concat([next_period_sample, city_encoded], axis=1)
                
                # S'assurer que toutes les colonnes du modèle sont présentes
                for col in X.columns:
                    if col not in city_features.columns:
                        city_features[col] = 0
                
                # Réorganiser les colonnes pour correspondre à l'ordre du modèle
                city_features = city_features[X.columns]
                
                # Prédire la consommation
                pred_conso = model.predict(city_features)[0]
            else:
                # Si pas de modèle, utiliser la moyenne pour cette ville
                pred_conso = df_current[df_current['Ville'] == city]['Conso'].mean()
            
            # Obtenir la consommation actuelle moyenne pour cette ville
            current_conso = df_current[df_current['Ville'] == city]['Conso'].mean()
            
            # Ajouter au dataframe des prédictions
            city_predictions.append({
                'Ville': city,
                f'Consommation {current_period_name} ({current_year})': current_conso,
                f'Prédiction {next_period_name} ({next_period_year})': pred_conso,
                'Variation (%)': ((pred_conso - current_conso) / current_conso * 100) if current_conso > 0 else 0
            })
        except Exception as e:
            st.error(f"Erreur lors de la prédiction pour {city}: {str(e)}")
    
    # Vérifier si nous avons des prédictions
    if not city_predictions:
        st.warning("⚠️ Impossible de générer des prédictions par ville.")
        return None, None, None
    
    # Créer un dataframe avec les prédictions
    city_pred_df = pd.DataFrame(city_predictions)
    
    # Trier par consommation prédite (décroissante)
    city_pred_df = city_pred_df.sort_values(f'Prédiction {next_period_name} ({next_period_year})', ascending=False)
    
    # Afficher le tableau des prédictions
    st.table(city_pred_df.style.format({
        f'Consommation {current_period_name} ({current_year})': '{:.2f}',
        f'Prédiction {next_period_name} ({next_period_year})': '{:.2f}',
        'Variation (%)': '{:.2f}'
    }))
    
    # Visualiser les prédictions par ville
    fig = go.Figure()
    
    # Ajouter les valeurs actuelles
    fig.add_trace(
        go.Bar(
            x=city_pred_df['Ville'],
            y=city_pred_df[f'Consommation {current_period_name} ({current_year})'],
            name=f'Consommation {current_period_name} ({current_year})',
            marker_color='blue'
        )
    )
    
    # Ajouter les valeurs prédites
    fig.add_trace(
        go.Bar(
            x=city_pred_df['Ville'],
            y=city_pred_df[f'Prédiction {next_period_name} ({next_period_year})'],
            name=f'Prédiction {next_period_name} ({next_period_year})',
            marker_color='red'
        )
    )
    
    # Mettre à jour la mise en page
    fig.update_layout(
        title=f"Comparaison de la consommation actuelle et prédite par ville",
        xaxis_title="Ville",
        yaxis_title="Consommation",
        barmode='group',
        height=500
    )
    
    # Afficher le graphique
    st.plotly_chart(fig)
    
    # Ajouter des interprétations détaillées
    st.write("### 📝 Interprétation des résultats par ville")
    
    # Identifier les villes avec les plus fortes hausses et baisses
    city_pred_df['Variation_abs'] = city_pred_df['Variation (%)'].abs()
    top_increasing = city_pred_df[city_pred_df['Variation (%)'] > 0].sort_values('Variation (%)', ascending=False).head(3)
    top_decreasing = city_pred_df[city_pred_df['Variation (%)'] < 0].sort_values('Variation (%)', ascending=True).head(3)
    
    # Tendances principales
    st.write("#### Tendances principales")
    
    # Calculer des statistiques
    avg_variation = city_pred_df['Variation (%)'].mean()
    num_increasing = len(city_pred_df[city_pred_df['Variation (%)'] > 0])
    num_decreasing = len(city_pred_df[city_pred_df['Variation (%)'] < 0])
    num_stable = len(city_pred_df[(city_pred_df['Variation (%)'] >= -1) & (city_pred_df['Variation (%)'] <= 1)])
    
    # Afficher la tendance globale
    if avg_variation > 3:
        st.write(f"🔺 **Tendance globale à la hausse** : En moyenne, les villes devraient connaître une augmentation de {avg_variation:.1f}% de leur consommation d'eau.")
    elif avg_variation < -3:
        st.write(f"🔻 **Tendance globale à la baisse** : En moyenne, les villes devraient connaître une diminution de {abs(avg_variation):.1f}% de leur consommation d'eau.")
    else:
        st.write(f"↔️ **Tendance globale stable** : En moyenne, les villes devraient connaître une variation limitée de {avg_variation:.1f}% de leur consommation d'eau.")
    
    st.write(f"- {num_increasing} villes présentent une tendance à la hausse")
    st.write(f"- {num_decreasing} villes présentent une tendance à la baisse")
    st.write(f"- {num_stable} villes présentent une consommation relativement stable (variation entre -1% et +1%)")
    
    # Villes à surveiller
    st.write("#### Villes à surveiller particulièrement")
    
    if not top_increasing.empty:
        st.write("**Villes avec la plus forte augmentation prévue :**")
        for i, row in top_increasing.iterrows():
            st.write(f"🔺 **{row['Ville']}** : +{row['Variation (%)']:.1f}% (de {row[f'Consommation {current_period_name} ({current_year})']:.1f} à {row[f'Prédiction {next_period_name} ({next_period_year})']:.1f})")
    
    if not top_decreasing.empty:
        st.write("**Villes avec la plus forte diminution prévue :**")
        for i, row in top_decreasing.iterrows():
            st.write(f"🔻 **{row['Ville']}** : {row['Variation (%)']:.1f}% (de {row[f'Consommation {current_period_name} ({current_year})']:.1f} à {row[f'Prédiction {next_period_name} ({next_period_year})']:.1f})")
    
    # Facteurs explicatifs potentiels
    st.write("#### Facteurs explicatifs potentiels")
    st.write("Les variations de consommation entre villes peuvent s'expliquer par plusieurs facteurs :")
    st.write("- **Facteurs démographiques** : évolution de la population, tourisme saisonnier")
    st.write("- **Facteurs économiques** : activité industrielle, développement commercial")
    st.write("- **Facteurs climatiques** : variations locales des conditions météorologiques")
    st.write("- **Facteurs infrastructurels** : état du réseau, travaux de maintenance, fuites")
    
    # Recommandations
    st.write("#### Recommandations")
    st.write("Sur la base de ces prédictions, nous recommandons de :")
    st.write("1. **Adapter les ressources** en fonction des tendances identifiées par ville")
    if not top_increasing.empty:
        st.write(f"2. **Anticiper une demande accrue** dans les villes en forte hausse, notamment {', '.join(top_increasing['Ville'].head(2).tolist())}")
    if not top_decreasing.empty:
        st.write(f"3. **Investiguer les causes** des baisses importantes dans certaines villes comme {', '.join(top_decreasing['Ville'].head(2).tolist())}")
    st.write("4. **Optimiser la distribution** en fonction des variations géographiques de la demande")
    st.write("5. **Établir un suivi spécifique** pour les villes présentant des variations atypiques")

    return model, cities, city_pred_df

# Fonction pour la prédiction par type de client
def predict_by_client_type(df, prediction_timeframe, selected_year):
    st.write(f"## 📈 Prédiction de consommation par type de client pour {prediction_timeframe}")
    
    # Déterminer la période en fonction du type de prédiction
    if "mois" in prediction_timeframe.lower():
        period_column = 'mois_debut'
        x_label = 'Mois'
        time_unit = 'mois'
    elif "trimestre" in prediction_timeframe.lower():
        period_column = 'trimestre_debut'
        x_label = 'Trimestre'
        time_unit = 'trimestre'
    elif "semestre" in prediction_timeframe.lower():
        period_column = 'semestre_debut'
        x_label = 'Semestre'
        time_unit = 'semestre'
    elif "année" in prediction_timeframe.lower() or "annee" in prediction_timeframe.lower():
        period_column = 'annee_debut'
        x_label = 'Année'
        time_unit = 'année'
    else:
        # Par défaut, utiliser le mois si le format n'est pas reconnu
        st.info(f"Format de période '{prediction_timeframe}' non reconnu, utilisation du mois par défaut")
        period_column = 'mois_debut'
        x_label = 'Mois'
        time_unit = 'mois'
    
    # Utiliser l'année sélectionnée
    current_year = selected_year
    
    # Déterminer la période actuelle et la période suivante à prédire
    if time_unit == 'mois':
        available_months = sorted(df[df['annee_debut'] == current_year]['mois_debut'].unique())
        if not available_months:
            # MODIFICATION: Si pas de données pour l'année sélectionnée, utiliser toutes les années
            available_months = sorted(df['mois_debut'].unique())
            if not available_months:
                st.warning(f"⚠️ Pas de données pour le mois")
                return None, None, None
                
            # Utiliser le dernier mois disponible dans toutes les données
            current_period = available_months[-1]
            st.info(f"Utilisation des données de toutes les années disponibles pour le mois {current_period}")
        else:
            current_period = available_months[-1]  # Dernier mois disponible
            
        next_period = current_period + 1 if current_period < 12 else 1
        next_period_year = current_year if next_period > current_period else current_year + 1
        current_period_name = calendar.month_name[current_period]
        next_period_name = calendar.month_name[next_period]
    
    elif time_unit == 'trimestre':
        available_quarters = sorted(df[df['annee_debut'] == current_year]['trimestre_debut'].unique())
        if not available_quarters:
            # MODIFICATION: Si pas de données pour l'année sélectionnée, utiliser toutes les années
            available_quarters = sorted(df['trimestre_debut'].unique())
            if not available_quarters:
                st.warning(f"⚠️ Pas de données pour le trimestre")
                return None, None, None
                
            # Utiliser le dernier trimestre disponible dans toutes les données
            current_period = available_quarters[-1]
            st.info(f"Utilisation des données de toutes les années disponibles pour le trimestre {current_period}")
        else:
            current_period = available_quarters[-1]  # Dernier trimestre disponible
            
        next_period = current_period + 1 if current_period < 4 else 1
        next_period_year = current_year if next_period > current_period else current_year + 1
        current_period_name = f"Q{current_period}"
        next_period_name = f"Q{next_period}"
    
    elif time_unit == 'semestre':
        available_semesters = sorted(df[df['annee_debut'] == current_year]['semestre_debut'].unique())
        if not available_semesters:
            # MODIFICATION: Si pas de données pour l'année sélectionnée, utiliser toutes les années
            available_semesters = sorted(df['semestre_debut'].unique())
            if not available_semesters:
                st.warning(f"⚠️ Pas de données pour le semestre")
                return None, None, None
                
            # Utiliser le dernier semestre disponible dans toutes les données
            current_period = available_semesters[-1]
            st.info(f"Utilisation des données de toutes les années disponibles pour le semestre {current_period}")
        else:
            current_period = available_semesters[-1]  # Dernier semestre disponible
            
        next_period = 2 if current_period == 1 else 1
        next_period_year = current_year if next_period > current_period else current_year + 1
        current_period_name = f"S{current_period}"
        next_period_name = f"S{next_period}"
    
    else:  # année
        current_period = current_year
        next_period = current_period + 1
        next_period_year = next_period
        current_period_name = str(current_period)
        next_period_name = str(next_period)
    
    # Filtrer les données pour la période actuelle
    df_current = df[(df[period_column] == current_period) & (df['annee_debut'] == current_year)]
    
    # MODIFICATION: Si pas assez de données pour la période actuelle, essayer d'utiliser toutes les données de cette période
    if len(df_current) < 5 or len(df_current['TypeClient'].unique()) < 2:
        st.info(f"Pas assez de données pour {time_unit} {current_period_name} de l'année {current_year}. Utilisation de toutes les années disponibles.")
        df_current = df[df[period_column] == current_period]
        
        # Si toujours pas assez de données
        if len(df_current) < 5 or len(df_current['TypeClient'].unique()) < 2:
            st.warning(f"⚠️ Toujours pas assez de données pour {time_unit} {current_period_name}. Utilisation de toutes les données disponibles.")
            df_current = df.copy()
    
    # Vérifier s'il y a suffisamment de types de clients et de données après les tentatives d'élargissement
    if len(df_current['TypeClient'].unique()) < 2:
        st.warning("⚠️ Pas assez de types de clients différents pour créer un modèle. Impossible de faire des prédictions par type de client.")
        return None, None, None
    
    if len(df_current) < 2:
        st.warning(f"⚠️ Pas assez de données pour entraîner un modèle fiable.")
        return None, None, None
    
    # Réinitialiser les indices
    df_current = df_current.reset_index(drop=True)
    
    try:
        # Encoder les types de client (one-hot encoding)
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        client_types_encoded = encoder.fit_transform(df_current[['TypeClient']])
        
        # Créer un dataframe avec les types de client encodés
        client_types_df = pd.DataFrame(
            client_types_encoded,
            columns=[f"type_{client_type}" for client_type in encoder.categories_[0][1:]]
        )
        
        # Concaténer avec les autres caractéristiques
        X = pd.concat([df_current[[period_column]], client_types_df], axis=1)
        y = df_current['Conso'].values
        








        # Créer et entraîner le modèle
        model = LinearRegression()
        model.fit(X, y)
        
        # Faire des prédictions pour la période actuelle
        y_pred = model.predict(X)










        
        # Évaluer le modèle
        metrics = evaluate_model(y, y_pred)
    except Exception as e:
        st.warning(f"⚠️ Erreur lors de l'entraînement du modèle: {str(e)}")
        st.info("Utilisation d'une approche simplifiée")
        # Approche simplifiée en cas d'erreur
        metrics = {
            'MSE': np.nan,
            'RMSE': np.nan,
            'MAE': np.nan,
            'R²': np.nan,
            'MAPE (%)': np.nan
        }
        # Continuer avec un modèle simpliste
        model = None
    
    # Afficher les métriques si un modèle a été créé
    # if model is not None:
    #     st.write(f"## 📊 Performance sur les données de {time_unit} {current_period_name} ({current_year})")
    #     display_metrics(metrics)


    if model is not None:
        st.write(f"## 📊 Performance sur les données de {time_unit} {current_period_name} ({current_year})")
        display_metrics(metrics)
        
        # NOUVEAU: Ajouter l'analyse des résidus
        st.write("## 📈 Analyse des résidus du modèle")
        
        try:
            # Utiliser la fonction plot_residuals pour afficher les graphiques des résidus
            plot_residuals(y, y_pred, title=f"Analyse des résidus pour la prédiction par type de client")
        except Exception as e:
            st.warning(f"Impossible d'afficher l'analyse des résidus: {str(e)}")

    
    # Prédiction pour chaque type de client pour la période suivante
    st.write(f"## 👥 Prédiction de consommation par type de client pour {time_unit} {next_period_name} ({next_period_year})")
    
    # Obtenir la liste des types de client uniques
    client_types = df_current['TypeClient'].unique()
    
    # Préparer un dataframe pour les prédictions par type de client
    client_type_predictions = []
    
    # Pour chaque type de client, prédire la consommation
    for client_type in client_types:
        try:
            if model is not None:
                # Créer un échantillon pour ce type de client pour la période suivante
                next_period_sample = pd.DataFrame({
                    period_column: [next_period]  # Utiliser la période suivante
                })
                
                # Encoder le type de client
                type_encoded = pd.DataFrame(columns=[f"type_{ct}" for ct in encoder.categories_[0][1:]])
                for col in type_encoded.columns:
                    type_encoded[col] = [1 if col == f"type_{client_type}" and client_type != encoder.categories_[0][0] else 0]
                
                # Concaténer les caractéristiques
                type_features = pd.concat([next_period_sample, type_encoded], axis=1)
                
                # S'assurer que toutes les colonnes du modèle sont présentes
                for col in X.columns:
                    if col not in type_features.columns:
                        type_features[col] = 0
                
                # Réorganiser les colonnes pour correspondre à l'ordre du modèle
                type_features = type_features[X.columns]
                
                # Prédire la consommation
                pred_conso = model.predict(type_features)[0]
            else:
                # Si pas de modèle, utiliser la moyenne pour ce type de client
                pred_conso = df_current[df_current['TypeClient'] == client_type]['Conso'].mean()
            
            # Obtenir la consommation actuelle moyenne pour ce type de client
            current_conso = df_current[df_current['TypeClient'] == client_type]['Conso'].mean()
            
            # Ajouter au dataframe des prédictions
            client_type_predictions.append({
                'Type de Client': client_type,
                f'Consommation {current_period_name} ({current_year})': current_conso,
                f'Prédiction {next_period_name} ({next_period_year})': pred_conso,
                'Variation (%)': ((pred_conso - current_conso) / current_conso * 100) if current_conso > 0 else 0
            })
        except Exception as e:
            st.error(f"Erreur lors de la prédiction pour {client_type}: {str(e)}")
    
    # Vérifier si nous avons des prédictions
    if not client_type_predictions:
        st.warning("⚠️ Impossible de générer des prédictions par type de client.")
        return None, None, None
    
    # Créer un dataframe avec les prédictions
    client_type_pred_df = pd.DataFrame(client_type_predictions)
    
    # Trier par consommation prédite (décroissante)
    client_type_pred_df = client_type_pred_df.sort_values(f'Prédiction {next_period_name} ({next_period_year})', ascending=False)
    
    # Afficher le tableau des prédictions
    st.table(client_type_pred_df.style.format({
        f'Consommation {current_period_name} ({current_year})': '{:.2f}',
        f'Prédiction {next_period_name} ({next_period_year})': '{:.2f}',
        'Variation (%)': '{:.2f}'
    }))
    
    # Visualiser les prédictions par type de client
    fig = go.Figure()
    
    # Ajouter les valeurs actuelles
    fig.add_trace(
        go.Bar(
            x=client_type_pred_df['Type de Client'],
            y=client_type_pred_df[f'Consommation {current_period_name} ({current_year})'],
            name=f'Consommation {current_period_name} ({current_year})',
            marker_color='blue'
        )
    )
    
    # Ajouter les valeurs prédites
    fig.add_trace(
        go.Bar(
            x=client_type_pred_df['Type de Client'],
            y=client_type_pred_df[f'Prédiction {next_period_name} ({next_period_year})'],
            name=f'Prédiction {next_period_name} ({next_period_year})',
            marker_color='red'
        )
    )
    
    # Mettre à jour la mise en page
    fig.update_layout(
        title=f"Comparaison de la consommation actuelle et prédite par type de client",
        xaxis_title="Type de Client",
        yaxis_title="Consommation",
        barmode='group',
        height=500
    )
    
    # Afficher le graphique
    st.plotly_chart(fig)
    
    # Ajouter des interprétations détaillées
    st.write("### 📝 Interprétation des résultats par type de client")
    
    # Identifier les types de clients avec les plus fortes hausses et baisses
    client_type_pred_df['Variation_abs'] = client_type_pred_df['Variation (%)'].abs()
    top_increasing = client_type_pred_df[client_type_pred_df['Variation (%)'] > 0].sort_values('Variation (%)', ascending=False).head(2)
    top_decreasing = client_type_pred_df[client_type_pred_df['Variation (%)'] < 0].sort_values('Variation (%)', ascending=True).head(2)
    
    # Tendances principales
    st.write("#### Tendances principales")
    
    # Calculer des statistiques
    avg_variation = client_type_pred_df['Variation (%)'].mean()
    num_increasing = len(client_type_pred_df[client_type_pred_df['Variation (%)'] > 0])
    num_decreasing = len(client_type_pred_df[client_type_pred_df['Variation (%)'] < 0])
    num_stable = len(client_type_pred_df[(client_type_pred_df['Variation (%)'] >= -1) & (client_type_pred_df['Variation (%)'] <= 1)])
    
    # Afficher la tendance globale
    if avg_variation > 3:
        st.write(f"🔺 **Tendance globale à la hausse** : En moyenne, les types de clients devraient connaître une augmentation de {avg_variation:.1f}% de leur consommation d'eau.")
    elif avg_variation < -3:
        st.write(f"🔻 **Tendance globale à la baisse** : En moyenne, les types de clients devraient connaître une diminution de {abs(avg_variation):.1f}% de leur consommation d'eau.")
    else:
        st.write(f"↔️ **Tendance globale stable** : En moyenne, les types de clients devraient connaître une variation limitée de {avg_variation:.1f}% de leur consommation d'eau.")
    
    st.write(f"- {num_increasing} types de clients présentent une tendance à la hausse")
    st.write(f"- {num_decreasing} types de clients présentent une tendance à la baisse")
    st.write(f"- {num_stable} types de clients présentent une consommation relativement stable (variation entre -1% et +1%)")
    
    # Types de clients à surveiller
    st.write("#### Types de clients à surveiller particulièrement")
    
    if not top_increasing.empty:
        st.write("**Types de clients avec la plus forte augmentation prévue :**")
        for i, row in top_increasing.iterrows():
            st.write(f"🔺 **{row['Type de Client']}** : +{row['Variation (%)']:.1f}% (de {row[f'Consommation {current_period_name} ({current_year})']:.1f} à {row[f'Prédiction {next_period_name} ({next_period_year})']:.1f})")
    
    if not top_decreasing.empty:
        st.write("**Types de clients avec la plus forte diminution prévue :**")
        for i, row in top_decreasing.iterrows():
            st.write(f"🔻 **{row['Type de Client']}** : {row['Variation (%)']:.1f}% (de {row[f'Consommation {current_period_name} ({current_year})']:.1f} à {row[f'Prédiction {next_period_name} ({next_period_year})']:.1f})")
    
    # Facteurs explicatifs potentiels
    st.write("#### Facteurs explicatifs potentiels")
    st.write("Les variations de consommation entre types de clients peuvent s'expliquer par plusieurs facteurs :")
    st.write("- **Facteurs économiques** : évolution de l'activité des entreprises, saisonnalité des besoins")
    st.write("- **Facteurs comportementaux** : changements dans les habitudes de consommation")
    st.write("- **Facteurs structurels** : évolution du nombre de clients dans chaque catégorie")
    st.write("- **Facteurs réglementaires** : nouvelles normes ou tarifications affectant certaines catégories")
    
    # Implications commerciales
    st.write("#### Implications commerciales")
    st.write("Ces prédictions peuvent avoir les implications suivantes pour la stratégie commerciale :")
    
    # Déterminer le type de client avec la plus forte consommation prédite
    if len(client_type_pred_df) > 0:
        top_consumer = client_type_pred_df.iloc[0]['Type de Client']
        st.write(f"- Les clients de type **{top_consumer}** représentent le segment avec la plus forte consommation prévue")
    
    # Recommandations basées sur les variations
    if num_increasing > num_decreasing:
        st.write("- La **hausse générale** de consommation suggère une opportunité d'optimisation des services et tarifs")
    elif num_decreasing > num_increasing:
        st.write("- La **baisse générale** de consommation suggère un besoin de fidélisation et de programmes d'incitation")
    
    # Recommandations
    st.write("#### Recommandations")
    st.write("Sur la base de ces prédictions, nous recommandons de :")
    st.write("1. **Adapter l'offre de services** en fonction des tendances par type de client")
    
    if not top_increasing.empty:
        st.write(f"2. **Préparer les ressources** pour répondre à l'augmentation de la demande des clients de type {', '.join(top_increasing['Type de Client'].head(1).tolist())}")
    
    if not top_decreasing.empty:
        st.write(f"3. **Développer des programmes de fidélisation** pour les segments en déclin comme {', '.join(top_decreasing['Type de Client'].head(1).tolist())}")
    
    st.write("4. **Optimiser la tarification** en fonction de l'élasticité de la demande par segment")
    st.write("5. **Mettre en place un suivi commercial** spécifique pour les types de clients présentant des variations importantes")
    
    return model, client_types, client_type_pred_df
# Mettre à jour la fonction dashboard pour inclure les nouvelles prédictions
def prediction_dashboard(df_filtered, selected_user, selected_year, prediction_timeframe):
    """
    Cette fonction intègre les modèles de prédiction dans l'interface principale,
    prenant en compte les modifications apportées aux fonctions de prédiction.
    
    Args:
        df_filtered: DataFrame filtré selon les critères de l'utilisateur
        selected_user: Utilisateur sélectionné
        selected_year: Année sélectionnée
        prediction_timeframe: Période à prédire (Mois prochain, Trimestre prochain, etc.)
    """
    # Titre de la section
    st.write("Cette analyse utilise des modèles de régression linéaire pour prédire la consommation d'eau pour la période future.")
    
    # Afficher l'information sur l'utilisateur et l'année sélectionnée
    if selected_user != "All users":
        st.write(f"**👤 Utilisateur:** {selected_user}")
    
    # Afficher l'année sélectionnée
    st.write(f"**📅 Année de référence:** {selected_year}")
    
    # Vérifier si nous avons des données disponibles
    if df_filtered.empty:
        st.warning("⚠️ Aucune donnée disponible avec les filtres actuels. Essayez d'élargir vos filtres.")
        return
    
    # Utiliser les mêmes onglets que dans le dashboard principal
    tab1, tab2, tab3 = st.tabs(["📅 Par Période", "🏙️ Par Ville", "👥 Par Type de Client"])
    
    with tab1:
        # Prédiction par période - utilise directement le prediction_timeframe
        predict_by_period(df_filtered, prediction_timeframe, prediction_timeframe, selected_year)
    
    with tab2:
        # Prédiction par ville - utilise directement le prediction_timeframe
        predict_by_city(df_filtered, prediction_timeframe, selected_year)
    
    with tab3:
        # Prédiction par type de client - utilise directement le prediction_timeframe
        predict_by_client_type(df_filtered, prediction_timeframe, selected_year)
