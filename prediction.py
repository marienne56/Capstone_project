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
from faker_consumption_script import generate_synthetic_data
import calendar
from datetime import datetime


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

def load_synthetic_data(num_records=5000):
    """
    Loads synthetic data generated with Faker
    
    Args:
        num_records: Number of records to generate
        
    Returns:
        DataFrame with synthetic data
    """
    try:
        # Generate synthetic data
        cities = ["Yamoussoukro", "San Pedro", "Jacqueville", "Abidjan", "Man", 
                 "Korhogo", "Séguela", "Katiola", "Boundiali", "Bouaké"]
        client_types = ["Residential", "Commercial", "Industrial"]
        
        # Generate the data
        df = generate_synthetic_data(
            num_records=num_records,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2025, 12, 31),
            cities=cities,
            client_types=client_types
        )
        
        # Convert date columns
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
        
        # For more precise analysis, consider the end period
        df['annee_fin'] = df['periodeFin'].dt.year
        df['trimestre_fin'] = df['periodeFin'].apply(get_quarter)
        df['semestre_fin'] = df['periodeFin'].apply(get_semester)
        
        # Calculate billing duration in days
        if 'duree_facturation' not in df.columns:
            df['duree_facturation'] = (df['periodeFin'] - df['periodeDebut']).dt.days
        
        # Calculate average consumption per day
        df['conso_par_jour'] = df['Conso'] / df['duree_facturation'].replace(0, 1)
        
        return df
    except Exception as e:
        st.error(f"Error generating synthetic data: {e}")
        import traceback
        st.error(traceback.format_exc())
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
    selected_year = st.sidebar.selectbox("year", sorted(df['annee_debut'].unique(), reverse=True))
    # selected_city = st.sidebar.selectbox("Ville", ["All cities"] + sorted(df['Ville'].unique()))
    # selected_client_type = st.sidebar.selectbox("Type de Client", ["All types"] + sorted(df['TypeClient'].unique()))
    # selected_payment_status = st.sidebar.selectbox("Statut de Paiement", ["All statuses"] + sorted(df['StatutPaiement'].unique()))
    
    #Options de prédiction basées sur la période
    # st.sidebar.markdown("## Options de Prédiction")
    # prediction_timeframe = st.sidebar.radio(
    #     "Prédire pour:",
    #      ["Mois prochain", "Trimestre prochain", "Semestre prochain", "year prochaine"]
    #  )
    
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



    """
    Fonction de diagnostic pour déterminer si le problème vient des données ou du modèle.
    Compare différents modèles et analyse la structure des données.
    
    Args:
        df_train: DataFrame contenant les données d'entraînement
        model_type: Type de modèle par défaut à utiliser
    
    Returns:
        dict: Résultats du diagnostic
    """
    st.write("## 🔍 Diagnostic approfondi: Données vs Modèle")
    
    # 1. ANALYSE DE LA STRUCTURE DES DONNÉES
    st.write("### Structure des données")
    
    # 1.1 Taille de l'échantillon
    st.write(f"Nombre total d'observations: **{len(df_train)}**")
    n_features = len(df_train.columns) - 1  # exclure la variable cible
    st.write(f"Nombre de caractéristiques potentielles: **{n_features}**")
    
    # Règle empirique: au moins 10-20 observations par paramètre du modèle
    min_recommended_samples = n_features * 20
    if len(df_train) < min_recommended_samples:
        st.error(f"⚠️ **Données insuffisantes**: Il est recommandé d'avoir au moins {min_recommended_samples} observations pour {n_features} caractéristiques.")
    else:
        st.success(f"✅ **Données suffisantes**: Le nombre d'observations est adéquat pour le nombre de caractéristiques.")
    
    # 1.2 Analyse des variables catégorielles
    cat_vars = [col for col in df_train.columns if df_train[col].dtype == 'object']
    for cat_var in cat_vars:
        value_counts = df_train[cat_var].value_counts()
        st.write(f"Distribution de {cat_var}:")
        
        # Nombre de valeurs uniques et distribution
        st.write(f"- Nombre de catégories uniques: {len(value_counts)}")
        
        # Identifier les catégories avec peu d'observations
        low_count_categories = value_counts[value_counts < 5].index.tolist()
        if low_count_categories:
            st.warning(f"- ⚠️ Catégories avec moins de 5 observations: {', '.join(low_count_categories)}")
    
    # 1.3 Analyse de la variable cible
    target_var = 'Conso'
    st.write(f"### Analyse de la variable cible '{target_var}'")
    
    # Statistiques descriptives
    target_stats = df_train[target_var].describe()
    st.write("Statistiques descriptives:")
    st.write(target_stats)
    
    # Coefficient de variation (mesure de dispersion relative)
    cv = target_stats['std'] / target_stats['mean'] * 100
    st.write(f"Coefficient de variation: {cv:.2f}%")
    
    if cv > 100:
        st.warning("⚠️ **Forte dispersion**: Le coefficient de variation > 100% indique une grande hétérogénéité des données.")
    
    # Visualiser la distribution
    try:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df_train[target_var],
            nbinsx=30,
            marker_color='blue'
        ))
        fig.update_layout(
            title="Distribution de la variable cible",
            xaxis_title=target_var,
            yaxis_title="Fréquence",
            height=400
        )
        st.plotly_chart(fig)
        
        # Test de normalité
        from scipy import stats
        stat, p_value = stats.shapiro(df_train[target_var])
        st.write(f"Test de normalité Shapiro-Wilk: p-value = {p_value:.6f}")
        if p_value < 0.05:
            st.info("La distribution n'est pas normale (p < 0.05). Une transformation pourrait améliorer les performances du modèle.")
    except Exception as e:
        st.error(f"Erreur lors de la visualisation: {str(e)}")
    
    # 2. COMPARAISON DE DIFFÉRENTS MODÈLES
    st.write("### Comparaison de différents modèles")
    
    # Préparation des données
    try:
        # Supposons que nous avons des variables catégorielles et numériques
        X_processed = pd.get_dummies(df_train.drop(['Conso'], axis=1), drop_first=True)
        y = df_train['Conso']
        
        # Créer un pipeline avec prétraitement standard et différents modèles
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.svm import SVR
        from sklearn.model_selection import cross_val_score
        
        # Différents modèles à tester
        models = {
            "Régression linéaire": LinearRegression(),
            "Ridge (régularisation L2)": Ridge(alpha=1.0),
            "Lasso (régularisation L1)": Lasso(alpha=0.1),
            "ElasticNet (L1+L2)": ElasticNet(alpha=0.1, l1_ratio=0.5),
            "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        }
        
        # Résultats de la validation croisée
        cv_results = {}
        
        for name, model in models.items():
            try:
                # Utiliser la validation croisée à 5 plis
                cv_scores = cross_val_score(model, X_processed, y, cv=5, scoring='r2')
                mean_score = cv_scores.mean()
                std_score = cv_scores.std()
                cv_results[name] = {
                    'mean_r2': mean_score,
                    'std_r2': std_score,
                    'cv_scores': cv_scores
                }
            except Exception as model_err:
                st.warning(f"Impossible d'évaluer le modèle {name}: {str(model_err)}")
        
        # Afficher les résultats
        results_df = pd.DataFrame({
            'Modèle': list(cv_results.keys()),
            'R² moyen': [cv_results[m]['mean_r2'] for m in cv_results],
            'Écart-type R²': [cv_results[m]['std_r2'] for m in cv_results]
        })
        
        # Trier par R² moyen décroissant
        results_df = results_df.sort_values('R² moyen', ascending=False)
        
        st.table(results_df.style.format({
            'R² moyen': '{:.4f}',
            'Écart-type R²': '{:.4f}'
        }))
        
        # Vérifier si tous les modèles ont de mauvaises performances
        best_r2 = results_df['R² moyen'].max()
        if best_r2 < 0.3:
            st.error("🚨 **Problème de données détecté**: Tous les modèles ont des performances faibles (R² < 0.3), ce qui suggère fortement un problème avec les données plutôt qu'avec le choix du modèle.")
        elif best_r2 < 0.6:
            st.warning("⚠️ **Performances limitées**: Tous les modèles ont des performances modérées. Les données pourraient être intrinsèquement difficiles à modéliser.")
        else:
            st.success(f"✅ **Modèle approprié trouvé**: Le meilleur modèle ({results_df.iloc[0]['Modèle']}) a un R² moyen de {best_r2:.4f}.")
        
        # Analyser la différence entre les meilleurs et les pires modèles
        r2_range = results_df['R² moyen'].max() - results_df['R² moyen'].min()
        if r2_range > 0.1:
            st.info(f"📊 **Sensibilité au type de modèle**: Grande variation de performance entre les modèles (ΔR² = {r2_range:.4f}). Le choix du modèle est important.")
        else:
            st.info(f"📊 **Faible sensibilité au type de modèle**: Tous les modèles ont des performances similaires (ΔR² = {r2_range:.4f}). Cela suggère une limite inhérente aux données.")
        
    except Exception as e:
        st.error(f"Erreur lors de la comparaison des modèles: {str(e)}")
    
    # 3. COURBE D'APPRENTISSAGE
    st.write("### Courbe d'apprentissage")
    st.write("Analyse de l'évolution des performances en fonction de la taille de l'échantillon d'entraînement")
    
    try:
        from sklearn.model_selection import learning_curve
        
        # Utiliser le meilleur modèle identifié ou celui spécifié par défaut
        if 'results_df' in locals() and len(results_df) > 0:
            best_model_name = results_df.iloc[0]['Modèle']
            best_model = models[best_model_name]
        else:
            best_model = Ridge(alpha=1.0)
            best_model_name = "Ridge"
        
        st.write(f"Modèle utilisé pour la courbe d'apprentissage: **{best_model_name}**")
        
        # Calculer la courbe d'apprentissage
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes, train_scores, test_scores = learning_curve(
            best_model, X_processed, y, 
            train_sizes=train_sizes, cv=5, scoring='r2',
            n_jobs=-1
        )
        
        # Calculer moyennes et écart-types
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Créer le graphique
        fig = go.Figure()
        
        # Données d'entraînement avec zone d'incertitude
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=train_mean,
            mode='lines+markers',
            name='Score d\'entraînement',
            line=dict(color='blue'),
            marker=dict(size=8)
        ))
        
        # Ajouter la zone d'incertitude pour l'entraînement
        fig.add_trace(go.Scatter(
            x=np.concatenate([train_sizes, train_sizes[::-1]]),
            y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
            fill='toself',
            fillcolor='rgba(0, 0, 255, 0.1)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            showlegend=False
        ))
        
        # Données de validation avec zone d'incertitude
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=test_mean,
            mode='lines+markers',
            name='Score de validation (CV)',
            line=dict(color='red'),
            marker=dict(size=8)
        ))
        
        # Ajouter la zone d'incertitude pour la validation
        fig.add_trace(go.Scatter(
            x=np.concatenate([train_sizes, train_sizes[::-1]]),
            y=np.concatenate([test_mean + test_std, (test_mean - test_std)[::-1]]),
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.1)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            showlegend=False
        ))
        
        # Mise en forme du graphique
        fig.update_layout(
            title="Courbe d'apprentissage",
            xaxis_title="Taille de l'échantillon d'entraînement",
            yaxis_title="Score R²",
            height=500
        )
        
        st.plotly_chart(fig)
        
        # Analyser la courbe d'apprentissage
        gap = train_mean[-1] - test_mean[-1]
        
        if gap > 0.3:
            st.warning("⚠️ **Surapprentissage détecté**: Grand écart entre les scores d'entraînement et de validation. Le modèle mémorise les données au lieu de généraliser.")
        
        # Vérifier si les performances augmentent encore avec plus de données
        slope = (test_mean[-1] - test_mean[-2]) / (train_sizes[-1] - train_sizes[-2])
        
        if slope > 0.01:
            st.info("📈 **Besoin de plus de données**: La courbe de validation continue d'augmenter. Collecter plus de données pourrait améliorer les performances.")
        else:
            st.info("📊 **Plateau atteint**: La courbe de validation s'aplatit. Ajouter plus de données n'améliorera probablement pas beaucoup les performances.")
        
        # Vérifier le score final
        final_score = test_mean[-1]
        if final_score < 0.3:
            st.error("🚨 **Performances limitées**: Même avec toutes les données disponibles, le modèle n'atteint pas des performances satisfaisantes (R² < 0.3).")
    
    except Exception as e:
        st.error(f"Erreur lors de la génération de la courbe d'apprentissage: {str(e)}")
    
    # 4. CONCLUSION
    st.write("### 📋 Conclusion du diagnostic")
    
    # Compiler les indices pour déterminer si le problème vient des données ou du modèle
    data_issues = []
    model_issues = []
    
    # Évaluer les problèmes potentiels de données
    if 'low_count_categories' in locals() and low_count_categories:
        data_issues.append("Catégories avec peu d'observations")
    
    if 'cv' in locals() and cv > 100:
        data_issues.append("Forte dispersion de la variable cible")
    
    if 'p_value' in locals() and p_value < 0.05:
        data_issues.append("Distribution non normale de la variable cible")
    
    if 'best_r2' in locals() and best_r2 < 0.3 and r2_range < 0.1:
        data_issues.append("Tous les modèles ont des performances similairement faibles")
    
    if 'slope' in locals() and slope > 0.01:
        data_issues.append("Besoin de plus de données selon la courbe d'apprentissage")
    
    if 'min_recommended_samples' in locals() and len(df_train) < min_recommended_samples:
        data_issues.append(f"Nombre d'observations insuffisant ({len(df_train)} < {min_recommended_samples} recommandés)")
    
    # Évaluer les problèmes potentiels de modèle
    if 'gap' in locals() and gap > 0.3:
        model_issues.append("Surapprentissage (écart important entre entraînement et validation)")
    
    if 'r2_range' in locals() and r2_range > 0.1:
        model_issues.append("Grande variabilité des performances entre les modèles")
    
    # Afficher le diagnostic final
    if len(data_issues) > len(model_issues):
        st.error("🚨 **Conclusion principale: Problème de données**")
        st.write("Le diagnostic suggère que les problèmes de performance sont principalement liés aux données:")
        for issue in data_issues:
            st.write(f"- {issue}")
            
        st.write("Recommandations:")
        st.write("1. **Collecter plus de données**, particulièrement pour les catégories sous-représentées")
        st.write("2. **Explorer des transformations** de la variable cible (log, racine carrée)")
        st.write("3. **Considérer des variables explicatives supplémentaires** qui pourraient mieux expliquer la variance")
    elif len(model_issues) > len(data_issues):
        st.warning("⚠️ **Conclusion principale: Problème de modèle**")
        st.write("Le diagnostic suggère que les problèmes de performance sont principalement liés au modèle:")
        for issue in model_issues:
            st.write(f"- {issue}")
            
        st.write("Recommandations:")
        st.write("1. **Essayer un modèle plus adapté** parmi ceux testés")
        st.write("2. **Ajuster les hyperparamètres** pour réduire le surapprentissage")
        st.write("3. **Considérer des techniques de régularisation** plus adaptées")
    else:
        st.info("📊 **Conclusion mixte: Problèmes de données et de modèle**")
        st.write("Le diagnostic suggère une combinaison de facteurs:")
        if data_issues:
            st.write("Problèmes de données:")
            for issue in data_issues:
                st.write(f"- {issue}")
        if model_issues:
            st.write("Problèmes de modèle:")
            for issue in model_issues:
                st.write(f"- {issue}")
    
    return {
        "data_issues": data_issues,
        "model_issues": model_issues,
        "best_model": best_model_name if 'best_model_name' in locals() else None,
        "best_r2": best_r2 if 'best_r2' in locals() else None
    }


def run_model_diagnostics_ridge(df_train, alpha=1.0):
    """
    Simplified diagnostic function that only uses the Ridge model
    and determines if the problem comes from the data.
    
    Args:
        df_train: DataFrame containing training data
        alpha: Regularization parameter for Ridge
    
    Returns:
        dict: Diagnostic results
    """
    st.write("## 🔍 Ridge Model Diagnostic")
    
    # 1. DATA STRUCTURE ANALYSIS
    #st.write("### Data Structure")
    
    # Display data types for debugging
    #st.write("Data types in DataFrame:")
    dtypes_df = pd.DataFrame({'Colonne': df_train.columns, 'Type': df_train.dtypes})
    st.table(dtypes_df)
    
    # 1.1 Sample size
    #st.write(f"Total number of observations: **{len(df_train)}**")
    n_features = len(df_train.columns) - 1  # exclude target variable
    #st.write(f"Number of potential features: **{n_features}**")
    
    # Empirical rule: at least 10-20 observations per model parameter
    min_recommended_samples = n_features * 10
    if len(df_train) < min_recommended_samples:
        st.error(f"⚠️ **Insufficient data**: It is recommended to have at least {min_recommended_samples} observations for {n_features} features.")
    else:
        st.success(f"✅ **Sufficient data**: The number of observations is adequate for the number of features.")
    
    # 1.3 Target variable analysis
    target_var = 'Conso'
    if 'Conso_capped' in df_train.columns:
        target_var = 'Conso_capped'  # Use capped version if available
    
    #st.write(f"### Analysis of target variable '{target_var}'")
    
    # Descriptive statistics
    target_stats = df_train[target_var].describe()
    # st.write("Descriptive statistics:")
    # st.write(target_stats)
    
    # Coefficient of variation (measure of relative dispersion)
    cv = target_stats['std'] / target_stats['mean'] * 100
    # st.write(f"Coefficient of variation: {cv:.2f}%")
    
    #if cv > 100:
        #st.warning("⚠️ **High dispersion**: Coefficient of variation > 100% indicates high data heterogeneity.")
    
    # 2. DATA PREPARATION FOR RIDGE MODEL
    #st.write("### Data Preparation")
    
    try:
        # Create a dataframe for analysis without the target variable
        X_df = df_train.drop([col for col in df_train.columns if col in ['Conso', 'Conso_capped']], axis=1)
        
        # STEP 1: Date processing
        # Identify and process datetime columns
        datetime_cols = X_df.select_dtypes(include=['datetime64']).columns.tolist()
        if datetime_cols:
            #st.write(f"Datetime columns detected: {datetime_cols}")
            #st.write("Converting dates to numeric features...")
            
            for col in datetime_cols:
                X_df[f'{col}_year'] = X_df[col].dt.year
                X_df[f'{col}_month'] = X_df[col].dt.month
                X_df[f'{col}_day'] = X_df[col].dt.day
                # Remove original datetime column
                X_df = X_df.drop(col, axis=1)
        else:
            st.write("No datetime columns detected.")
        
        # STEP 2: Categorical variables processing
        cat_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            #st.write(f"Categorical columns detected: {cat_cols}")
            X_encoded = pd.get_dummies(X_df, columns=cat_cols, drop_first=True)
        else:
            X_encoded = X_df
            #st.write("No categorical columns detected.")
        
        # STEP 3: Check that there are no more non-numeric columns
        non_numeric_cols = X_encoded.select_dtypes(exclude=['int', 'float']).columns.tolist()
        if non_numeric_cols:
            #st.warning(f"Remaining non-numeric columns: {non_numeric_cols}")
            
            # Convert these columns to numeric or remove them
            for col in non_numeric_cols:
                #st.write(f"Converting column {col} to numeric type...")
                X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce')
            
            # Remove columns that couldn't be converted
            X_encoded = X_encoded.select_dtypes(include=['int', 'float'])
        
        # Check final dimensions
        X_processed = X_encoded
        y = df_train[target_var]
        
        #st.write(f"Data dimensions after preprocessing: X = {X_processed.shape}, y = {y.shape}")
        
        # 3. RIDGE MODEL EVALUATION
        st.write("### Ridge Model Evaluation")
        
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import cross_val_score, train_test_split
        
        # Create a pipeline with data standardization
        ridge_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=alpha))
        ])
        
        # Train/test split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        ridge_pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = ridge_pipeline.predict(X_train)
        y_pred_test = ridge_pipeline.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        train_metrics = {
            'MSE': mean_squared_error(y_train, y_pred_train),
            'RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'MAE': mean_absolute_error(y_train, y_pred_train),
            'R²': r2_score(y_train, y_pred_train)
        }
        
        test_metrics = {
            'MSE': mean_squared_error(y_test, y_pred_test),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'MAE': mean_absolute_error(y_test, y_pred_test),
            'R²': r2_score(y_test, y_pred_test)
        }
        
        # Display metrics
        st.write("#### Training metrics:")
        metrics_df_train = pd.DataFrame({
            'Métrique': list(train_metrics.keys()),
            'Valeur': list(train_metrics.values())
        })
        st.table(metrics_df_train.style.format({
            'Valeur': '{:.4f}'
        }))
        
        st.write("#### Test metrics:")
        metrics_df_test = pd.DataFrame({
            'Métrique': list(test_metrics.keys()),
            'Valeur': list(test_metrics.values())
        })
        st.table(metrics_df_test.style.format({
            'Valeur': '{:.4f}'
        }))
        
        # Analyze the gap between train and test
        r2_train = train_metrics['R²']
        r2_test = test_metrics['R²']
        r2_gap = r2_train - r2_test
        
        # Calculate relative error (MAE/mean)
        avg_target = y.mean()
        mae_test = test_metrics['MAE']
        mae_relative = (mae_test / avg_target) * 100
        
        # Performance evaluation
        st.write("### Performance Analysis")
        
        if r2_test < 0.2:
            st.error(f"🚨 **Very low performance**: R² = {r2_test:.4f}. The model explains less than 20% of the variance.")
        elif r2_test < 0.4:
            st.warning(f"⚠️ **Low performance**: R² = {r2_test:.4f}. The model explains less than 40% of the variance.")
        elif r2_test < 0.6:
            st.info(f"ℹ️ **Moderate performance**: R² = {r2_test:.4f}. The model explains between 40% and 60% of the variance.")
        else:
            st.success(f"✅ **Good performance**: R² = {r2_test:.4f}. The model explains more than 60% of the variance.")
        
        # Analyze the gap between train and test
        if r2_gap > 0.2:
            st.warning(f"⚠️ **Possible overfitting**: Gap of {r2_gap:.4f} between training and test R².")
        else:
            st.success(f"✅ **No overfitting**: Acceptable gap of {r2_gap:.4f} between training and test R².")
        
        # Analyze relative error
        if mae_relative > 50:
            st.error(f"🚨 **Very high error**: Relative MAE = {mae_relative:.2f}% of the mean value.")
        elif mae_relative > 30:
            st.warning(f"⚠️ **High error**: Relative MAE = {mae_relative:.2f}% of the mean value.")
        elif mae_relative > 20:
            st.info(f"ℹ️ **Moderate error**: Relative MAE = {mae_relative:.2f}% of the mean value.")
        else:
            st.success(f"✅ **Low error**: Relative MAE = {mae_relative:.2f}% of the mean value.")
        
        
        # Warning if very low performance
        if r2_test < 0.2:
            st.write("\n### ⚠️ Important note:")
            st.write(f"With an R² of {r2_test:.4f}, current predictions are **not reliable**. It is recommended to use these predictions with extreme caution.")
        
        return {
            # "data_issues": data_issues,
            # "model_issues": model_issues,
            "r2_train": r2_train,
            "r2_test": r2_test,
            "mae_relative": mae_relative
        }
    except Exception as e:
        st.error("Error during data analysis:")
        st.code(str(e))
        return {"error": str(e)}
def predict_by_period(df, period_type, prediction_timeframe, selected_year):
    """
    Fonction de prédiction par période avec entraînement uniquement sur les données les plus récentes
    pour réduire les variations extrêmes, incluant métriques d'évaluation et analyse des résidus.
    """
    # Adapter le titre en fonction du type de prédiction
    #st.write(f"## 📈 Prédiction de consommation pour {prediction_timeframe}")
    
    # Déterminer la colonne de période en fonction du type de prédiction
    if "month" in prediction_timeframe.lower():
        period_column = 'mois_debut' 
        x_label = 'Mois'
        time_unit = 'month'
    elif "quarter" in prediction_timeframe.lower():
        period_column = 'trimestre_debut'
        x_label = 'Trimestre'
        time_unit = 'quarter'
    elif "semester" in prediction_timeframe.lower():
        period_column = 'semestre_debut'
        x_label = 'semester'
        time_unit = 'semester'
    elif "year" in prediction_timeframe.lower() or "year" in prediction_timeframe.lower():
        period_column = 'annee_debut'
        x_label = 'year'
        time_unit = 'year'
    else:
        st.warning(f"Period format'{prediction_timeframe}' not recognized.")
        return None, None, None, None
    
    # Afficher les valeurs moyennes par période pour analyse
    agg_data = df.groupby(period_column)['Conso'].mean().reset_index()
    #st.write("Consommation moyenne par période:")
    st.table(agg_data)
    
    # Déterminer la période actuelle et la période suivante
    current_year = selected_year
    
    if time_unit == 'month':
        available_months = sorted(df[df['annee_debut'] == current_year]['mois_debut'].unique())
        if not available_months:
            st.error(f"⚠️ Not enought data for  {current_year}")
            return None, None, None, None
        
        current_period = available_months[-1]
        next_period = current_period + 1 if current_period < 12 else 1
        next_period_year = current_year if next_period > current_period else current_year + 1
        current_period_name = calendar.month_name[current_period]
        next_period_name = calendar.month_name[next_period]
    
    elif time_unit == 'quarter':
        available_quarters = sorted(df[df['annee_debut'] == current_year]['trimestre_debut'].unique())
        if not available_quarters:
            st.error(f"⚠️ Not enough data for {current_year}")
            return None, None, None, None
        
        current_period = available_quarters[-1]
        next_period = current_period + 1 if current_period < 4 else 1
        next_period_year = current_year if next_period > current_period else current_year + 1
        current_period_name = f"Q{current_period}"
        next_period_name = f"Q{next_period}"
    
    elif time_unit == 'semester':
        available_semesters = sorted(df[df['annee_debut'] == current_year]['semestre_debut'].unique())
        if not available_semesters:
            
            st.error(f" ⚠️ No data for year {current_year}")
                     
            return None, None, None, None
        
        current_period = available_semesters[-1]
        next_period = 2 if current_period == 1 else 1
        next_period_year = current_year if next_period > current_period else current_year + 1
        current_period_name = f"S{current_period}"
        next_period_name = f"S{next_period}"
    
    else:  # year
        current_period = current_year
        next_period = current_period + 1
        next_period_year = next_period
        current_period_name = str(current_period)
        next_period_name = str(next_period)
    
    # st.write(f"Période actuelle: {current_period_name} {current_year}")
    # st.write(f"Période à prédire: {next_period_name} {next_period_year}")
    
    # NOUVELLE APPROCHE: Sélection très ciblée des données d'entraînement
    # Pour chaque type de période, utiliser uniquement les périodes les plus récentes
    
    if time_unit == 'month':
        # Pour les mois, utiliser seulement les 12 derniers mois
        #st.write("### Approche d'entraînement: 12 derniers mois seulement")
        
        # Créer une liste de tuples (year, mois) pour toutes les données
        date_tuples = [(row['annee_debut'], row['mois_debut']) for _, row in df.iterrows()]
        unique_date_tuples = list(set(date_tuples))
        
        # Trier par year puis par mois, en ordre décroissant
        unique_date_tuples.sort(reverse=True)
        
        # Prendre les 12 plus récentes (ou moins si pas assez de données)
        recent_tuples = unique_date_tuples[:min(12, len(unique_date_tuples))]
        
        # Filtrer le dataframe pour ces dates
        df_train = df[df.apply(lambda row: (row['annee_debut'], row['mois_debut']) in recent_tuples, axis=1)].copy()
        
        #st.write(f"Utilisation de {len(df_train)} enregistrements sur {len(df)} (des {len(recent_tuples)} derniers mois)")
    
    elif time_unit == 'quarter':
        # Pour les trimestres, utiliser seulement les 4 derniers trimestres
        #st.write("### Approche d'entraînement: 4 derniers trimestres seulement")
        
        # Créer une liste de tuples (year, trimestre) pour toutes les données
        date_tuples = [(row['annee_debut'], row['trimestre_debut']) for _, row in df.iterrows()]
        unique_date_tuples = list(set(date_tuples))
        
        # Trier par year puis par trimestre, en ordre décroissant
        unique_date_tuples.sort(reverse=True)
        
        # Prendre les 4 plus récents (ou moins si pas assez de données)
        recent_tuples = unique_date_tuples[:min(4, len(unique_date_tuples))]
        
        # Filtrer le dataframe pour ces dates
        df_train = df[df.apply(lambda row: (row['annee_debut'], row['trimestre_debut']) in recent_tuples, axis=1)].copy()
        
        #st.write(f"Utilisation de {len(df_train)} enregistrements sur {len(df)} (des {len(recent_tuples)} derniers trimestres)")
        
    elif time_unit == 'semester':
        # Pour les semestres, utiliser seulement les 2 derniers semestres
        #st.write("### Approche d'entraînement: 2 derniers semestres seulement")
        
        # Créer une liste de tuples (year, semestre) pour toutes les données
        date_tuples = [(row['annee_debut'], row['semestre_debut']) for _, row in df.iterrows()]
        unique_date_tuples = list(set(date_tuples))
        
        # Trier par year puis par semestre, en ordre décroissant
        unique_date_tuples.sort(reverse=True)
        
        # Prendre les 2 plus récents (ou moins si pas assez de données)
        recent_tuples = unique_date_tuples[:min(2, len(unique_date_tuples))]
        
        # Filtrer le dataframe pour ces dates
        df_train = df[df.apply(lambda row: (row['annee_debut'], row['semestre_debut']) in recent_tuples, axis=1)].copy()
        
        #st.write(f"Utilisation de {len(df_train)} enregistrements sur {len(df)} (des {len(recent_tuples)} derniers semestres)")
    
    else:  # year
        # Pour les years, utiliser toutes les years précédentes
        #st.write("### Approche d'entraînement: Toutes les years disponibles")
        
        # Obtenir toutes les years disponibles
        all_years = sorted(df['annee_debut'].unique())
        
        # Filtrer pour ne garder que les years précédentes ou égales à l'year actuelle
        previous_years = [year for year in all_years if year <= current_year]
        
        # Utiliser toutes les données des years disponibles jusqu'à l'year actuelle
        df_train = df[df['annee_debut'].isin(previous_years)].copy()
        
        #st.write(f"Utilisation de {len(df_train)} enregistrements sur {len(df)} (de toutes les {len(previous_years)} years jusqu'à {current_year})")
    
    # Vérifier qu'il y a suffisamment de données
    min_required_data = 4  # Nombre minimum de points de données
    if len(df_train) < min_required_data:
        #st.warning(f"⚠️ Peu de données disponibles ({len(df_train)} points). Les prédictions peuvent être moins fiables.")
        # Si pas assez de données, utiliser toutes les données disponibles
        df_train = df.copy()
        #st.write(f"Retour à l'utilisation de toutes les données ({len(df_train)} enregistrements)")
    
    # Préparer X_train (caractéristiques) et y_train (valeurs cibles)
    X_train = df_train[period_column].values.reshape(-1, 1)
    y_train = df_train['Conso'].values
    
    st.write("### Model Training")
    
    if time_unit == 'month':
        # Pour les mois : Ridge avec contrainte modérée (la sélection de données est déjà restrictive)
        alpha = 5.0
        model = Ridge(alpha=alpha)
        #st.write(f"Modèle utilisé: Ridge avec alpha={alpha}")
    elif time_unit == 'quarter':
        # Pour les trimestres : Ridge avec contrainte légère
        alpha = 3.0
        model = Ridge(alpha=alpha)
        #st.write(f"Modèle utilisé: Ridge avec alpha={alpha}")
    elif time_unit == 'semester':
        # Pour les semestres : Ridge avec contrainte très légère
        alpha = 1.0
        model = Ridge(alpha=alpha)
        #st.write(f"Modèle utilisé: Ridge avec alpha={alpha}")
    else:
        # Pour les years : Régression linéaire simple + petite régularisation
        alpha = 0.5
        model = Ridge(alpha=alpha)
        #st.write(f"Modèle utilisé: Ridge avec alpha={alpha}")
    
    # Entraîner le modèle
    model.fit(X_train, y_train)

    #Ajouter ce code
    if st.checkbox("View in-depth model diagnostics.", False, key="diagnostic_ridgee"):
        run_model_diagnostics_ridge(df_train)


    # Afficher les coefficients du modèle
    # st.write(f"Coefficient: {model.coef_[0]:.4f}")
    # st.write(f"Intercept: {model.intercept_:.4f}")
    # st.write(f"Équation: Consommation = {model.coef_[0]:.4f} × Période + {model.intercept_:.4f}")
    
    # Faire des prédictions pour la période actuelle (pour validation)
    y_pred_train = model.predict(X_train)
    
    # # Évaluer le modèle avec les métriques détaillées
    # metrics = evaluate_model(y_train, y_pred_train)
    
    # # Afficher les métriques d'évaluation
    #st.write("📊 Performance sur les données d'entraînement")
    # display_metrics(metrics)
    
    # Calculer l'erreur moyenne sur les données d'entraînement
    mean_error = np.mean(np.abs(y_train - y_pred_train))
    mean_error_pct = (mean_error / np.mean(y_train)) * 100 if np.mean(y_train) > 0 else 0
    
    #st.write(f"Erreur moyenne sur données d'entraînement: {mean_error:.2f} ({mean_error_pct:.2f}%)")
    
    # Ajouter l'analyse des résidus
    # st.write("## 📈 Analyse des résidus du modèle")
    # try:
    #     # Utiliser la fonction plot_residuals pour afficher les graphiques des résidus
    #     plot_residuals(y_train, y_pred_train, title=f"Analyse des résidus pour la prédiction par {time_unit}")
    # except Exception as e:
    #     st.warning(f"Impossible d'afficher l'analyse des résidus: {str(e)}")
    
    # Faire une prédiction pour la période suivante
    X_forecast = np.array([[next_period]])
    next_period_prediction = model.predict(X_forecast)[0]
    
    # S'assurer que la prédiction n'est pas négative
    next_period_prediction = max(0, next_period_prediction)
    
    # Calculer la consommation moyenne pour la période actuelle
    current_data = df[(df[period_column] == current_period) & (df['annee_debut'] == current_year)]
    
    if current_data.empty:
        st.warning(f"No data for {current_period_name} {current_year}. Use of overall average.")
        current_conso = df_train['Conso'].mean()
    else:
        current_conso = current_data['Conso'].mean()
    
    # Calcul de la variation
    if current_conso > 0:
        variation_pct = ((next_period_prediction - current_conso) / current_conso) * 100
    else:
        variation_pct = 0
    
    # Limiter les variations extrêmes pour les affichages
    display_variation = variation_pct
    max_reasonable_variation = 30.0  # Variation maximale considérée comme "raisonnable"
    
    # Afficher la prédiction
    st.write(f"### 📈 Prediction for {time_unit} {next_period_name} {f'({next_period_year})' if time_unit != 'year' else ''}")
    
    if abs(display_variation) > max_reasonable_variation:
        st.warning(f"⚠️ Large variation detected ({display_variation:.1f}%). This prediction may be less reliable and should be interpreted with caution.")
        st.info(f"Note: In an operational context, it would be advisable to limit this variation to  ±{max_reasonable_variation}%.")
 
        # Corriger la prédiction elle-même, pas seulement l'affichage
        direction = 1 if variation_pct > 0 else -1
        adjusted_variation = direction * max_reasonable_variation
        next_period_prediction = current_conso * (1 + adjusted_variation/100)
        display_variation = adjusted_variation
        variation_pct = adjusted_variation
        #st.write(f"Prediction adjusted to {next_period_prediction:.2f} to limit variation to ±{max_reasonable_variation}%")
        
    
    # Afficher la prédiction
    #st.write(f"### Prédiction pour {time_unit} {next_period_name} ({next_period_year})")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        delta = next_period_prediction - current_conso
        st.metric(
            label=f"Prediction for {next_period_name}",
            value=f"{next_period_prediction:.2f}",
            delta=f"{delta:.2f}"
        )
    with col2:
        st.metric(
            label=f"Current consumption",
            value=f"{current_conso:.2f}"
        )
    with col3:
        st.metric(
            label="Variation",
            value=f"{display_variation:.2f}%",
            delta=f"{delta:.2f}"
        )
    
    # # Créer un tableau pour comparer les valeurs
    # comparison_df = pd.DataFrame({
    #     'Périod': [f"{current_period_name} ({current_year})", f"{next_period_name} ({next_period_year})"],
    #     'Consumption': [current_conso, next_period_prediction],
    #     'Variation (%)': [0, display_variation]
    # })
    
    # st.table(comparison_df.style.format({
    #     'Consumption': '{:.2f}',
    #     'Variation (%)': '{:.2f}'
    # }))
    
    # Créer un graphique à barres (bar chart)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=["Historical data", "Prediction"],
            y=[current_conso, next_period_prediction],
            marker_color=['blue', 'red'],
            text=[f"{current_conso:.2f}", f"{next_period_prediction:.2f}"],
            textposition='auto'
        )
    )

    fig.update_layout(
        title=f"Comparison between current data and prediction for {next_period_name}",
        xaxis_title="Data type",
        yaxis_title="Consumption",
        height=500,
        barmode='group'
    )

    st.plotly_chart(fig)


    # Now, let's also predict the billing amount in FCFA
    st.write("### 💵 Billing Amount Prediction")

    # Prepare billing training data
    y_train_billing = df_train['MontFact'].values if 'MontFact' in df_train.columns else None

    if y_train_billing is not None:
        # Train a separate Ridge model for billing
        billing_model = Ridge(alpha=alpha)
        billing_model.fit(X_train, y_train_billing)
        
        # Predict billing for next period
        next_period_billing = billing_model.predict(X_forecast)[0]
        next_period_billing = max(0, next_period_billing)  # Ensure non-negative
        
        # Get current period billing
        current_billing_data = df[(df[period_column] == current_period) & (df['annee_debut'] == current_year)]
        
        if current_billing_data.empty:
            st.warning(f"No billing data for {current_period_name} {current_year}. Using overall average.")
            current_billing = df_train['MontFact'].mean() if 'MontFact' in df_train.columns else 0
        else:
            current_billing = current_billing_data['MontFact'].mean() if 'MontFact' in current_billing_data.columns else 0
        
        # Calculate the variation
        if current_billing > 0:
            billing_variation_pct = ((next_period_billing - current_billing) / current_billing) * 100
        else:
            billing_variation_pct = 0
        
        # Limit extreme variations for display
        display_billing_variation = billing_variation_pct
        
        if abs(display_billing_variation) > max_reasonable_variation:
            st.warning(f"⚠️ Large billing variation detected ({display_billing_variation:.1f}%). This prediction may be less reliable.")
            st.info(f"Note: In an operational context, it would be advisable to limit this variation to ±{max_reasonable_variation}%.")
            
            # Adjust the prediction to reasonable limits
            direction = 1 if billing_variation_pct > 0 else -1
            adjusted_billing_variation = direction * max_reasonable_variation
            next_period_billing = current_billing * (1 + adjusted_billing_variation/100)
            display_billing_variation = adjusted_billing_variation
            billing_variation_pct = adjusted_billing_variation
        
        # Display billing prediction
        col1, col2, col3 = st.columns(3)
        with col1:
            delta_billing = next_period_billing - current_billing
            st.metric(
                label=f"Predicted billing for {next_period_name}",
                value=f"{next_period_billing:.2f} FCFA",
                delta=f"{delta_billing:.2f}"
            )
        with col2:
            st.metric(
                label=f"Current billing",
                value=f"{current_billing:.2f} FCFA"
            )
        with col3:
            st.metric(
                label="Variation",
                value=f"{display_billing_variation:.2f}%",
                delta=f"{delta_billing:.2f}"
            )
        
        # Create a comparison bar chart
        fig_billing = go.Figure()
        fig_billing.add_trace(
            go.Bar(
                x=["Historical data", "Prediction"],
                y=[current_billing, next_period_billing],
                marker_color=['blue', 'red'],
                text=[f"{current_billing:.2f} FCFA", f"{next_period_billing:.2f} FCFA"],
                textposition='auto'
            )
        )

        fig_billing.update_layout(
            title=f"Comparison between current billing and prediction for {next_period_name}",
            xaxis_title="Data type",
            yaxis_title="Billing Amount (FCFA)",
            height=500,
            barmode='group'
        )

        st.plotly_chart(fig_billing)
    
    # Ajouter une interprétation des résultats
    st.write("### 📝 Interpretation of the Results")
    
    if abs(variation_pct) < 5:
        st.write(f"🔄 The prediction shows a **minimal variation** of {variation_pct:.1f}% pour {time_unit} {next_period_name} compared to the current data.")
    elif variation_pct > 15:
        st.write(f"🔺 The prediction shows a **significant increase** of {variation_pct:.1f}% pour {time_unit} {next_period_name} compared to the current data.")
    elif variation_pct > 5:
        st.write(f"↗️ The prediction shows a **moderate increase** of  {variation_pct:.1f}% pour {time_unit} {next_period_name} compared to the current data.")
    elif variation_pct < -15:
        st.write(f"🔻 The prediction shows a **significant decrease** of  {abs(variation_pct):.1f}% pour {time_unit} {next_period_name} compared to the current data.")
    else:
        st.write(f"↘️ The prediction shows a **slight decrease** of {abs(variation_pct):.1f}% pour {time_unit} {next_period_name} compared to the current data.")
    
    # Fiabilité de la prédiction
    st.write("#### Prediction Reliability")
    
    if len(df_train) < 10:
        st.write("⚠️ Prediction based on **limited data** (<10 points). Results should be interpreted with caution.")
    elif abs(variation_pct) > 30:
        st.write("⚠️ The **significant variation** suggests that the prediction should be interpreted with caution.")
    else:
        st.write("✅ The prediction seems reasonable given the available data.")
    
    return model, current_period, next_period, next_period_prediction
def predict_by_city(df, prediction_timeframe, selected_year):
    #st.write(f"## 📈 Prédiction de consommation par ville pour {prediction_timeframe}")
    
    # Déterminer la période en fonction du type de prédiction
    if "month" in prediction_timeframe.lower():
        period_column = 'mois_debut'
        x_label = 'Mois'
        time_unit = 'month'
    elif "quarter" in prediction_timeframe.lower():
        period_column = 'trimestre_debut'
        x_label = 'Trimestre'
        time_unit = 'quarter'
    elif "semester" in prediction_timeframe.lower():
        period_column = 'semestre_debut'
        x_label = 'Semestre'
        time_unit = 'semester'
    elif "year" in prediction_timeframe.lower() or "year" in prediction_timeframe.lower():
        period_column = 'annee_debut'
        x_label = 'year'
        time_unit = 'year'
    else:
        # Par défaut, utiliser le mois si le format n'est pas reconnu
        st.info(f"period format '{prediction_timeframe}' not recognized, using of the default month")
        period_column = 'mois_debut'
        x_label = 'month'
        time_unit = 'month'
    
    # Utiliser l'year sélectionnée
    current_year = selected_year
    
    # Déterminer la période actuelle (selon l'year sélectionnée) et la période suivante à prédire
    if time_unit == 'month':
        available_months = sorted(df[df['annee_debut'] == current_year]['mois_debut'].unique())
        if not available_months:
            # Si pas de données pour l'year sélectionnée, utiliser toutes les years
            available_months = sorted(df['mois_debut'].unique())
            if not available_months:
                st.warning(f"⚠️ no data for the monthe ")
                return None, None, None
                
            # Utiliser le dernier mois disponible dans toutes les données
            current_period = available_months[-1]
            st.info(f"Using data from all available years for the month {current_period}")
        else:
            current_period = available_months[-1]  # Dernier mois disponible
            
        next_period = current_period + 1 if current_period < 12 else 1
        next_period_year = current_year if next_period > current_period else current_year + 1
        current_period_name = calendar.month_name[current_period]
        next_period_name = calendar.month_name[next_period]
    
    elif time_unit == 'quarter':
        available_quarters = sorted(df[df['annee_debut'] == current_year]['trimestre_debut'].unique())
        if not available_quarters:
            # Si pas de données pour l'year sélectionnée, utiliser toutes les years
            available_quarters = sorted(df['trimestre_debut'].unique())
            if not available_quarters:
                st.warning(f"⚠️ No data for the quarter")
                return None, None, None
                
            # Utiliser le dernier trimestre disponible dans toutes les données
            current_period = available_quarters[-1]
            st.info(f"Using from all data from available years for the quarter{current_period}")
        else:
            current_period = available_quarters[-1]  # Dernier trimestre disponible
            
        next_period = current_period + 1 if current_period < 4 else 1
        next_period_year = current_year if next_period > current_period else current_year + 1
        current_period_name = f"Q{current_period}"
        next_period_name = f"Q{next_period}"
    
    elif time_unit == 'semester':
        available_semesters = sorted(df[df['annee_debut'] == current_year]['semestre_debut'].unique())
        if not available_semesters:
            # Si pas de données pour l'year sélectionnée, utiliser toutes les years
            available_semesters = sorted(df['semestre_debut'].unique())
            if not available_semesters:
                st.warning(f"⚠️ No data for the semester")
                return None, None, None
                
            # Utiliser le dernier semestre disponible dans toutes les données
            current_period = available_semesters[-1]
            st.info(f"Using from all data from the available years for the semester {current_period}")
        else:
            current_period = available_semesters[-1]  # Dernier semestre disponible
            
        next_period = 2 if current_period == 1 else 1
        next_period_year = current_year if next_period > current_period else current_year + 1
        current_period_name = f"S{current_period}"
        next_period_name = f"S{next_period}"
    
    else:  # year
        current_period = current_year
        next_period = current_period + 1
        next_period_year = next_period
        current_period_name = str(current_period)
        next_period_name = str(next_period)
    
    # AMÉLIORATION: Utiliser une approche similaire à celle de predict_by_period avec une sélection ciblée des données
    # Filtrer les données pour obtenir des données plus récentes pour un modèle plus pertinent
    #st.write("### Sélection des données d'entraînement")
    if time_unit == 'month':
        # Pour les mois, utiliser les 12 derniers mois pour chaque ville
        #st.write("Approche d'entraînement: 12 derniers mois de données par ville")
        
        # Créer une liste de tuples (year, mois) pour toutes les données
        date_tuples = [(row['annee_debut'], row['mois_debut']) for _, row in df.iterrows()]
        unique_date_tuples = list(set(date_tuples))
        
        # Trier par year puis par mois, en ordre décroissant
        unique_date_tuples.sort(reverse=True)
        
        # Prendre les 12 plus récentes (ou moins si pas assez de données)
        recent_tuples = unique_date_tuples[:min(12, len(unique_date_tuples))]
        
        # Filtrer le dataframe pour ces dates
        df_train = df[df.apply(lambda row: (row['annee_debut'], row['mois_debut']) in recent_tuples, axis=1)].copy()
        
        #st.write(f"Utilisation de {len(df_train)} enregistrements sur {len(df)} (des {len(recent_tuples)} derniers mois)")
    
    elif time_unit == 'quarter':
        # Pour les trimestres, utiliser les 4 derniers trimestres
        #st.write("Approche d'entraînement: 4 derniers trimestres de données par ville")
        
        date_tuples = [(row['annee_debut'], row['trimestre_debut']) for _, row in df.iterrows()]
        unique_date_tuples = list(set(date_tuples))
        
        unique_date_tuples.sort(reverse=True)
        recent_tuples = unique_date_tuples[:min(4, len(unique_date_tuples))]
        
        df_train = df[df.apply(lambda row: (row['annee_debut'], row['trimestre_debut']) in recent_tuples, axis=1)].copy()
        
        #st.write(f"Utilisation de {len(df_train)} enregistrements sur {len(df)} (des {len(recent_tuples)} derniers trimestres)")
    
    elif time_unit == 'semester':
        # Pour les semestres, utiliser les 2 derniers semestres
        #st.write("Approche d'entraînement: 2 derniers semestres de données par ville")
        
        date_tuples = [(row['annee_debut'], row['semestre_debut']) for _, row in df.iterrows()]
        unique_date_tuples = list(set(date_tuples))
        
        unique_date_tuples.sort(reverse=True)
        recent_tuples = unique_date_tuples[:min(2, len(unique_date_tuples))]
        
        df_train = df[df.apply(lambda row: (row['annee_debut'], row['semestre_debut']) in recent_tuples, axis=1)].copy()
        
        #st.write(f"Utilisation de {len(df_train)} enregistrements sur {len(df)} (des {len(recent_tuples)} derniers semestres)")
    
    else:  # year
        # Pour les years, utiliser toutes les years disponibles
        #st.write("Approche d'entraînement: Données de toutes les years disponibles")
        
        all_years = sorted(df['annee_debut'].unique())
        previous_years = [year for year in all_years if year <= current_year]
        
        df_train = df[df['annee_debut'].isin(previous_years)].copy()
        
        #st.write(f"Utilisation de {len(df_train)} enregistrements sur {len(df)} (de toutes les {len(previous_years)} years jusqu'à {current_year})")
    
    # Filtrer les données pour la période actuelle pour afficher les comparaisons
    df_current = df_train[(df_train[period_column] == current_period)]
    
    # Si pas assez de données pour la période actuelle, utiliser toutes les données d'entraînement
    if len(df_current) < 5 or len(df_current['Ville'].unique()) < 2:
        st.info(f"Not enough data for {time_unit} {current_period_name}. Using of selected training data.")
        df_current = df_train.copy()
    
    # Vérifier s'il y a suffisamment de villes et de données
    if len(df_train['Ville'].unique()) < 2:
        st.warning("⚠️ Not enough different cities to create a model. Impossible to make predictions by city.")
        return None, None, None
    
    if len(df_train) < 10:
        st.warning(f"⚠️ Few data available ({len(df_train)} points). Predictions can be less reliable.")
    
    # Réinitialiser les indices
    df_train = df_train.reset_index(drop=True)
    
    # AMÉLIORATION: Prétraitement des données pour gérer les valeurs extrêmes
    #st.write("### Prétraitement des données")
    
    # Détection et traitement des valeurs aberrantes (capping)
    Q1 = df_train['Conso'].quantile(0.25)
    Q3 = df_train['Conso'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Afficher les seuils
    #st.write(f"Seuil inférieur de capping: {lower_bound:.2f}")
    #st.write(f"Seuil supérieur de capping: {upper_bound:.2f}")
    
    # Nombre de valeurs aberrantes avant capping
    outliers_count = len(df_train[(df_train['Conso'] < lower_bound) | (df_train['Conso'] > upper_bound)])
    #st.write(f"Nombre de valeurs aberrantes détectées: {outliers_count} ({outliers_count/len(df_train)*100:.2f}% des données)")
    
    # Appliquer le capping
    df_train_capped = df_train.copy()
    df_train_capped['Conso_capped'] = df_train['Conso'].clip(lower=lower_bound, upper=upper_bound)
    
    # AMÉLIORATION: Ajouter des caractéristiques supplémentaires pour améliorer le modèle
    # 1. Ajouter des caractéristiques temporelles
    df_train_capped['period_value'] = df_train_capped[period_column]
    
    # 2. Ajouter la moyenne de consommation par ville (feature d'encodage de cible)
    city_means = df_train.groupby('Ville')['Conso'].mean().reset_index()
    city_means.rename(columns={'Conso': 'city_mean_conso'}, inplace=True)
    df_train_capped = pd.merge(df_train_capped, city_means, on='Ville', how='left')
    
    # 3. Ajouter la moyenne de consommation par période
    period_means = df_train.groupby(period_column)['Conso'].mean().reset_index()
    period_means.rename(columns={'Conso': 'period_mean_conso'}, inplace=True)
    df_train_capped = pd.merge(df_train_capped, period_means, on=period_column, how='left')
    
    # Encoder les villes (one-hot encoding)
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    cities_encoded = encoder.fit_transform(df_train_capped[['Ville']])
    
    # Créer un dataframe avec les villes encodées
    cities_df = pd.DataFrame(
        cities_encoded,
        columns=[f"ville_{city}" for city in encoder.categories_[0][1:]]
    )
    
    # AMÉLIORATION: Utiliser Ridge au lieu de LinearRegression pour la régularisation
    try:
        # Concaténer avec les autres caractéristiques
        X = pd.concat([
            df_train_capped[['period_value', 'city_mean_conso', 'period_mean_conso']], 
            cities_df
        ], axis=1)
        
        # Utiliser la version cappée de la consommation comme cible
        y = df_train_capped['Conso_capped'].values
        
        # AMÉLIORATION: Utiliser Ridge avec régularisation pour éviter le surajustement
        alpha_value = 1.0  # Paramètre de régularisation - ajuster selon les données
        model = Ridge(alpha=alpha_value)
        
        #st.write(f"Modèle utilisé: Ridge avec alpha={alpha_value} (régularisation)")
        model.fit(X, y)

        if st.checkbox("View in-depth model diagnostics", False):
            run_model_diagnostics_ridge(df_train)
        
        # Faire des prédictions pour les données d'entraînement (pour évaluer le modèle)
        y_pred = model.predict(X)
        
        # Évaluer le modèle
        #metrics = evaluate_model(y, y_pred)
        
        # # Ajouter l'analyse des coefficients du modèle
        # st.write("### Coefficients du modèle")
        # coef_df = pd.DataFrame({
        #     'Feature': X.columns,
        #     'Coefficient': model.coef_
        # })
        # coef_df = coef_df.sort_values('Coefficient', ascending=False)
        # st.write("Top 5 caractéristiques les plus influentes:")
        # st.table(coef_df.head(5))
        
    except Exception as e:
        st.warning(f"⚠️ Erreur lors de l'entraînement du modèle: {str(e)}")
        st.info("Model training error:")
        #Approche simplifiée en cas d'erreur
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
    #     st.write(f"## 📊 Performance sur les données d'entraînement")
    #     display_metrics(metrics)
        
    #     # Ajouter l'analyse des résidus
    #     st.write("## 📈 Analyse des résidus du modèle")
        
    #     try:
    #         # Utiliser la fonction plot_residuals pour afficher les graphiques des résidus
    #         plot_residuals(y, y_pred, title=f"Analyse des résidus pour la prédiction par ville")
            
    #         # AMÉLIORATION: Ajouter un graphique de comparaison valeurs réelles vs prédites
    #         st.write("### Comparaison entre valeurs réelles et prédites")
    #         comparison_data = pd.DataFrame({
    #             'Réel': y,
    #             'Prédit': y_pred,
    #             'Erreur': y - y_pred,
    #             'Erreur (%)': (y - y_pred) / y * 100 if np.any(y != 0) else np.zeros_like(y),
    #             'Ville': df_train_capped['Ville'].values,
    #             'Période': df_train_capped[period_column].values
    #         })
            
    #         fig = go.Figure()
    #         fig.add_trace(go.Scatter(
    #             x=comparison_data['Réel'],
    #             y=comparison_data['Prédit'],
    #             mode='markers',
    #             marker=dict(color='blue'),
    #             name='Consommation'
    #         ))
            
    #         # Ajouter une ligne de référence parfaite (y=x)
    #         min_val = min(comparison_data['Réel'].min(), comparison_data['Prédit'].min())
    #         max_val = max(comparison_data['Réel'].max(), comparison_data['Prédit'].max())
    #         fig.add_trace(go.Scatter(
    #             x=[min_val, max_val],
    #             y=[min_val, max_val],
    #             mode='lines',
    #             line=dict(color='red', dash='dash'),
    #             name='Prédiction parfaite'
    #         ))
            
    #         fig.update_layout(
    #             title='Valeurs réelles vs Prédites',
    #             xaxis_title='Consommation réelle',
    #             yaxis_title='Consommation prédite',
    #             height=500
    #         )
            
    #         st.plotly_chart(fig)
            
    #     except Exception as e:
    #         st.warning(f"Impossible d'afficher les analyses supplémentaires: {str(e)}")
    
    # Prédiction pour chaque ville pour la période suivante
    #st.write(f"## 🏙️ Prédiction de consommation par ville pour {time_unit} {next_period_name}")
    
    # Obtenir la liste des villes uniques
    cities = df_train_capped['Ville'].unique()
    
    # Préparer un dataframe pour les prédictions par ville
    city_predictions = []
    
    # Pour chaque ville, prédire la consommation
    for city in cities:
        try:
            if model is not None:
                # Créer un échantillon pour cette ville pour la période suivante
                next_period_sample = pd.DataFrame({
                    'period_value': [next_period],
                    'city_mean_conso': [city_means[city_means['Ville'] == city]['city_mean_conso'].values[0]],
                    'period_mean_conso': [period_means[period_means[period_column] == next_period]['period_mean_conso'].values[0] 
                                        if next_period in period_means[period_column].values 
                                        else period_means['period_mean_conso'].mean()]
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
                
                # AMÉLIORATION: Limiter les prédictions extrêmes
                # Utiliser les statistiques historiques pour borner les prédictions
                city_historical = df[df['Ville'] == city]
                if not city_historical.empty:
                    city_min = city_historical['Conso'].min()
                    city_max = city_historical['Conso'].max()
                    city_mean = city_historical['Conso'].mean()
                    
                    # Définir des limites raisonnables basées sur l'historique
                    lower_limit = max(0, city_min * 0.7)  # Ne pas aller en dessous de 70% du minimum historique
                    upper_limit = city_max * 1.3  # Ne pas aller au-dessus de 130% du maximum historique
                    
                    # Limiter la prédiction
                    if pred_conso < lower_limit:
                        #st.info(f"Prédiction ajustée pour {city}: {pred_conso:.2f} → {lower_limit:.2f} (limite inférieure)")
                        pred_conso = lower_limit
                    elif pred_conso > upper_limit:
                        #st.info(f"Prédiction ajustée pour {city}: {pred_conso:.2f} → {upper_limit:.2f} (limite supérieure)")
                        pred_conso = upper_limit
            else:
                # Si pas de modèle, utiliser la moyenne pour cette ville
                pred_conso = df_train_capped[df_train_capped['Ville'] == city]['Conso'].mean()
            
            # Obtenir la consommation actuelle moyenne pour cette ville
            current_city_data = df_current[df_current['Ville'] == city]
            if current_city_data.empty:
                current_conso = df_train_capped[df_train_capped['Ville'] == city]['Conso'].mean()
            else:
                current_conso = current_city_data['Conso'].mean()
            
            # Calculer la variation avec protection contre la division par zéro
            if current_conso > 0:
                variation_pct = ((pred_conso - current_conso) / current_conso * 100)
            else:
                variation_pct = 0
                
            # AMÉLIORATION: Limiter les variations extrêmes pour les affichages
            max_reasonable_variation = 30.0
            if abs(variation_pct) > max_reasonable_variation:
                #st.info(f"Variation extrême détectée pour {city}: {variation_pct:.1f}%. Ajustement à ±{max_reasonable_variation}%.")
                direction = 1 if variation_pct > 0 else -1
                variation_pct = direction * max_reasonable_variation
                pred_conso = current_conso * (1 + variation_pct/100)
            
            # Ajouter au dataframe des prédictions
            city_predictions.append({
                'Ville': city,
                f'Consommation {current_period_name} ({current_year})': current_conso,
                f'Prédiction {next_period_name} ({next_period_year})': pred_conso,
                'Variation (%)': variation_pct
            })
        except Exception as e:
            st.error(f"Error during the prediction {city}: {str(e)}")
    
    # Vérifier si nous avons des prédictions
    if not city_predictions:
        st.warning("⚠️ Impossible to generate predictions by city")
        return None, None, None
    
    # Créer un dataframe avec les prédictions
    city_pred_df = pd.DataFrame(city_predictions)
    
    # Trier par consommation prédite (décroissante)
    city_pred_df = city_pred_df.sort_values(f'Prédiction {next_period_name} ({next_period_year})', ascending=False)
    
    st.write(f"## 📈 Prediction for consumption by city for {prediction_timeframe}")
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
            name=f'Current Consumption',
            marker_color='blue'
        )
    )
    
    # Ajouter les valeurs prédites
    fig.add_trace(
        go.Bar(
            x=city_pred_df['Ville'],
            y=city_pred_df[f'Prédiction {next_period_name} ({next_period_year})'],
            name=f'Prediction {next_period_name} ({next_period_year})',
            marker_color='red'
        )
    )
    
    # Mettre à jour la mise en page
    fig.update_layout(
        title=f"Comparision between the current and predicted consumption by city",
        xaxis_title="City",
        yaxis_title="Consumption",
        barmode='group',
        height=500
    )
    
    # Afficher le graphique
    st.plotly_chart(fig)
    
    # Ajouter des interprétations détaillées (comme dans le code original)
    st.write("### 📝 Interpretation of result by city")
    
    # Identifier les villes avec les plus fortes hausses et baisses
    city_pred_df['Variation_abs'] = city_pred_df['Variation (%)'].abs()
    top_increasing = city_pred_df[city_pred_df['Variation (%)'] > 0].sort_values('Variation (%)', ascending=False).head(3)
    top_decreasing = city_pred_df[city_pred_df['Variation (%)'] < 0].sort_values('Variation (%)', ascending=True).head(3)
    
    # Tendances principales
    st.write("#### Main trends")
    
    # Calculer des statistiques
    avg_variation = city_pred_df['Variation (%)'].mean()
    num_increasing = len(city_pred_df[city_pred_df['Variation (%)'] > 0])
    num_decreasing = len(city_pred_df[city_pred_df['Variation (%)'] < 0])
    num_stable = len(city_pred_df[(city_pred_df['Variation (%)'] >= -1) & (city_pred_df['Variation (%)'] <= 1)])
    
    # Display overall trend
    if avg_variation > 3:
        st.write(f"🔺 **Overall upward trend**: On average, cities are expected to experience an increase of {avg_variation:.1f}% in their water consumption.")
    elif avg_variation < -3:
        st.write(f"🔻 **Overall downward trend**: On average, cities are expected to experience a decrease of {abs(avg_variation):.1f}% in their water consumption.")
    else:
        st.write(f"↔️ **Stable overall trend**: On average, cities should experience a limited variation of {avg_variation:.1f}% in their water consumption.")

    st.write(f"- {num_increasing} cities show an upward trend")
    st.write(f"- {num_decreasing} cities show a downward trend")
    st.write(f"- {num_stable} cities have relatively stable consumption (variation between -1% and +1%)")

    #SAME CONTENT AS THE ORIGINAL FOR THE FOLLOWING PART
    # Cities to monitor
    st.write("#### Cities to Monitor Closely")

    if not top_increasing.empty:
        st.write("**Cities with the highest expected increase:**")
        for i, row in top_increasing.iterrows():
            st.write(f"🔺 **{row['Ville']}**: +{row['Variation (%)']:.1f}% (from {row[f'Consommation {current_period_name} ({current_year})']:.1f} to {row[f'Prédiction {next_period_name} ({next_period_year})']:.1f})")

    if not top_decreasing.empty:
        st.write("**Cities with the highest expected decrease:**")
        for i, row in top_decreasing.iterrows():
            st.write(f"🔻 **{row['Ville']}**: {row['Variation (%)']:.1f}% (from {row[f'Consommation {current_period_name} ({current_year})']:.1f} to {row[f'Prédiction {next_period_name} ({next_period_year})']:.1f})")

        # Potential explanatory factors
    st.write("#### Potential Explanatory Factors")
    st.write("Variations in consumption between cities can be explained by several factors:")
    st.write("- **Demographic factors**: population growth, seasonal tourism")
    st.write("- **Economic factors**: industrial activity, commercial development")
    st.write("- **Climatic factors**: local variations in weather conditions")
    st.write("- **Infrastructure factors**: network condition, maintenance work, leaks")

    # Recommandations
    st.write("#### Recommendations")
    st.write("Based on these predictions, we recommend:")
    st.write("1. **Adjusting resources** according to the identified trends by city")
    if not top_increasing.empty:
        st.write(f"2. **Anticipating increased demand** in cities with strong growth, particularly {', '.join(top_increasing['Ville'].head(2).tolist())}")
    if not top_decreasing.empty:
        st.write(f"3. **Investigating causes** of significant decreases in some cities like {', '.join(top_decreasing['Ville'].head(2).tolist())}")
    st.write("4. **Optimizing distribution** based on geographic variations in demand")
    st.write("5. **Establishing specific monitoring** for cities with atypical variations")
    
    # AMÉLIORATION: Ajouter une section sur la fiabilité des prédictions
    #st.write("#### Fiabilité des prédictions")
    
    # Évaluer la fiabilité en fonction de plusieurs facteurs
    # if model is not None and 'R²' in metrics:
    #     r2 = metrics['R²']
    #     if r2 > 0.7:
    #         st.write("✅ **Fiabilité élevée** : Le modèle explique bien les variations dans les données (R² > 0.7)")
    #     elif r2 > 0.5:
    #         st.write("🟢 **Fiabilité modérée** : Le modèle capture une partie significative des variations (R² > 0.5)")
    #     elif r2 > 0.2:
    #         st.write("🟡 **Fiabilité limitée** : Le modèle ne capture qu'une partie modeste des variations (R² > 0.2)")
    #     else:
    #         st.write("🔴 **Fiabilité faible** : Le modèle explique peu les variations dans les données (R² < 0.2)")
    #         st.write("⚠️ Il est recommandé d'utiliser ces prédictions avec prudence et de les compléter avec d'autres méthodes d'analyse.")
    # else:
    #     st.write("⚠️ **Fiabilité indéterminée** : Impossible d'évaluer précisément la fiabilité du modèle")

    # Fiabilité de la prédiction
    st.write("#### Prediction Reliability")

    
    if len(df_train) < 10:
        st.write("⚠️ Prediction based on **limited data** (<10 points). The results should be interpreted with caution.")
    elif abs(variation_pct) > 20:
        st.write("⚠️ The **significant variation** indicates a prediction that should be interpreted carefully.")
    else:
        st.write("✅ The prediction seems reasonable given the available data.")

    
    # Factors influencing reliability
    st.write("**Factors Influencing Prediction Reliability:**")
    st.write("- Amount of available data: " + ("✅ Sufficient" if len(df_train) > 50 else "⚠️ Limited"))
    st.write("- Diversity of cities: " + ("✅ Good" if len(cities) > 5 else "⚠️ Limited"))
    st.write("- Regularity of temporal data: " + ("✅ Regular" if len(df_train) > len(cities) * 3 else "⚠️ Irregular"))
    st.write("- Presence of extreme values: " + ("⚠️ Significant" if outliers_count/len(df_train)*100 > 10 else "✅ Moderate"))
    return model, cities, city_pred_df
# Fonction pour la prédiction par type de client
def predict_by_client_type(df, prediction_timeframe, selected_year):
    #st.write(f"## 📈 Prédiction de consommation par type de client pour {prediction_timeframe}")
    
    # Déterminer la période en fonction du type de prédiction
    if "month" in prediction_timeframe.lower():
        period_column = 'mois_debut'
        x_label = 'month'
        time_unit = 'month'
    elif "quarter" in prediction_timeframe.lower():
        period_column = 'trimestre_debut'
        x_label = 'quarter'
        time_unit = 'quarter'
    elif "semester" in prediction_timeframe.lower():
        period_column = 'semestre_debut'
        x_label = 'Semestre'
        time_unit = 'semester'
    elif "year" in prediction_timeframe.lower() or "year" in prediction_timeframe.lower():
        period_column = 'annee_debut'
        x_label = 'year'
        time_unit = 'year'
    else:
        # Par défaut, utiliser le mois si le format n'est pas reconnu
        st.info(f"period format '{prediction_timeframe}' not recognized,using of the default month")
        period_column = 'mois_debut'
        x_label = 'Mois'
        time_unit = 'month'
    
    # Utiliser l'year sélectionnée
    current_year = selected_year
    
    # Déterminer la période actuelle (selon l'year sélectionnée) et la période suivante à prédire
    if time_unit == 'month':
        available_months = sorted(df[df['annee_debut'] == current_year]['mois_debut'].unique())
        if not available_months:
            # Si pas de données pour l'year sélectionnée, utiliser toutes les years
            available_months = sorted(df['mois_debut'].unique())
            if not available_months:
                st.warning(f"⚠️ No data for the month")
                return None, None, None
                
            # Utiliser le dernier mois disponible dans toutes les données
            current_period = available_months[-1]
            st.info(f"Using of all available years data for the month {current_period}")
        else:
            current_period = available_months[-1]  # Dernier mois disponible
            
        next_period = current_period + 1 if current_period < 12 else 1
        next_period_year = current_year if next_period > current_period else current_year + 1
        current_period_name = calendar.month_name[current_period]
        next_period_name = calendar.month_name[next_period]
    
    elif time_unit == 'quarter':
        available_quarters = sorted(df[df['annee_debut'] == current_year]['trimestre_debut'].unique())
        if not available_quarters:
            # Si pas de données pour l'year sélectionnée, utiliser toutes les years
            available_quarters = sorted(df['trimestre_debut'].unique())
            if not available_quarters:
                st.warning(f"⚠️ No data for the quarter")
                return None, None, None
                
            # Utiliser le dernier trimestre disponible dans toutes les données
            current_period = available_quarters[-1]
            st.info(f"Using of all available years data for the quarter{current_period}")
        else:
            current_period = available_quarters[-1]  # Dernier trimestre disponible
            
        next_period = current_period + 1 if current_period < 4 else 1
        next_period_year = current_year if next_period > current_period else current_year + 1
        current_period_name = f"Q{current_period}"
        next_period_name = f"Q{next_period}"
    
    elif time_unit == 'semester':
        available_semesters = sorted(df[df['annee_debut'] == current_year]['semestre_debut'].unique())
        if not available_semesters:
            # Si pas de données pour l'year sélectionnée, utiliser toutes les years
            available_semesters = sorted(df['semestre_debut'].unique())
            if not available_semesters:
                st.warning(f"⚠️ Not enough data for the semester")
                return None, None, None
                
            # Utiliser le dernier semestre disponible dans toutes les données
            current_period = available_semesters[-1]
            st.info(f"Using of all available years data for the semester {current_period}")
        else:
            current_period = available_semesters[-1]  # Dernier semestre disponible
            
        next_period = 2 if current_period == 1 else 1
        next_period_year = current_year if next_period > current_period else current_year + 1
        current_period_name = f"S{current_period}"
        next_period_name = f"S{next_period}"
    
    else:  # year
        current_period = current_year
        next_period = current_period + 1
        next_period_year = next_period
        current_period_name = str(current_period)
        next_period_name = str(next_period)
    
    # AMÉLIORATION: Utiliser une approche similaire à celle de predict_by_period avec une sélection ciblée des données
    # Filtrer les données pour obtenir des données plus récentes pour un modèle plus pertinent
    #st.write("### Sélection des données d'entraînement")
    if time_unit == 'month':
        # Pour les mois, utiliser les 12 derniers mois pour chaque type de client
        #st.write("Approche d'entraînement: 12 derniers mois de données par type de client")
        
        # Créer une liste de tuples (year, mois) pour toutes les données
        date_tuples = [(row['annee_debut'], row['mois_debut']) for _, row in df.iterrows()]
        unique_date_tuples = list(set(date_tuples))
        
        # Trier par year puis par mois, en ordre décroissant
        unique_date_tuples.sort(reverse=True)
        
        # Prendre les 12 plus récentes (ou moins si pas assez de données)
        recent_tuples = unique_date_tuples[:min(12, len(unique_date_tuples))]
        
        # Filtrer le dataframe pour ces dates
        df_train = df[df.apply(lambda row: (row['annee_debut'], row['mois_debut']) in recent_tuples, axis=1)].copy()
        
        #st.write(f"Utilisation de {len(df_train)} enregistrements sur {len(df)} (des {len(recent_tuples)} derniers mois)")
    
    elif time_unit == 'quarter':
        # Pour les trimestres, utiliser les 4 derniers trimestres
        #st.write("Approche d'entraînement: 4 derniers trimestres de données par type de client")
        
        date_tuples = [(row['annee_debut'], row['trimestre_debut']) for _, row in df.iterrows()]
        unique_date_tuples = list(set(date_tuples))
        
        unique_date_tuples.sort(reverse=True)
        recent_tuples = unique_date_tuples[:min(4, len(unique_date_tuples))]
        
        df_train = df[df.apply(lambda row: (row['annee_debut'], row['trimestre_debut']) in recent_tuples, axis=1)].copy()
        
        #st.write(f"Utilisation de {len(df_train)} enregistrements sur {len(df)} (des {len(recent_tuples)} derniers trimestres)")
    
    elif time_unit == 'semester':
        # Pour les semestres, utiliser les 2 derniers semestres
        #st.write("Approche d'entraînement: 2 derniers semestres de données par type de client")
        
        date_tuples = [(row['annee_debut'], row['semestre_debut']) for _, row in df.iterrows()]
        unique_date_tuples = list(set(date_tuples))
        
        unique_date_tuples.sort(reverse=True)
        recent_tuples = unique_date_tuples[:min(2, len(unique_date_tuples))]
        
        df_train = df[df.apply(lambda row: (row['annee_debut'], row['semestre_debut']) in recent_tuples, axis=1)].copy()
        
        #st.write(f"Utilisation de {len(df_train)} enregistrements sur {len(df)} (des {len(recent_tuples)} derniers semestres)")
    
    else:  # year
        # Pour les years, utiliser toutes les years disponibles
        #st.write("Approche d'entraînement: Données de toutes les years disponibles")
        
        all_years = sorted(df['annee_debut'].unique())
        previous_years = [year for year in all_years if year <= current_year]
        
        df_train = df[df['annee_debut'].isin(previous_years)].copy()
        
        #st.write(f"Utilisation de {len(df_train)} enregistrements sur {len(df)} (de toutes les {len(previous_years)} years jusqu'à {current_year})")
    
    # Filtrer les données pour la période actuelle pour afficher les comparaisons
    df_current = df_train[(df_train[period_column] == current_period)]
    
    # Si pas assez de données pour la période actuelle, utiliser toutes les données d'entraînement
    if len(df_current) < 5 or len(df_current['TypeClient'].unique()) < 2:
        st.info(f"Not enough data for  {time_unit} {current_period_name}. Use of selected training data.")
        df_current = df_train.copy()
    
    # Vérifier s'il y a suffisamment de types de clients et de données
    if len(df_train['TypeClient'].unique()) < 2:
        st.warning("⚠️ Not enough different customer types to create a model. Impossible to make predictions by customer type.")
        return None, None, None
    
    if len(df_train) < 10:
        st.warning(f"⚠️ Few available data ({len(df_train)} points). Predictions may be less reliable.")
    
    # Réinitialiser les indices
    df_train = df_train.reset_index(drop=True)
    
    # AMÉLIORATION: Prétraitement des données pour gérer les valeurs extrêmes
    #st.write("### Prétraitement des données")
    
    # Détection et traitement des valeurs aberrantes (capping)
    Q1 = df_train['Conso'].quantile(0.25)
    Q3 = df_train['Conso'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Afficher les seuils
    #st.write(f"Seuil inférieur de capping: {lower_bound:.2f}")
    #st.write(f"Seuil supérieur de capping: {upper_bound:.2f}")
    
    # Nombre de valeurs aberrantes avant capping
    outliers_count = len(df_train[(df_train['Conso'] < lower_bound) | (df_train['Conso'] > upper_bound)])
    #st.write(f"Nombre de valeurs aberrantes détectées: {outliers_count} ({outliers_count/len(df_train)*100:.2f}% des données)")
    
    # Appliquer le capping
    df_train_capped = df_train.copy()
    df_train_capped['Conso_capped'] = df_train['Conso'].clip(lower=lower_bound, upper=upper_bound)
    
    # AMÉLIORATION: Ajouter des caractéristiques supplémentaires pour améliorer le modèle
    # 1. Ajouter des caractéristiques temporelles
    df_train_capped['period_value'] = df_train_capped[period_column]
    
    # 2. Ajouter la moyenne de consommation par type de client (feature d'encodage de cible)
    client_means = df_train.groupby('TypeClient')['Conso'].mean().reset_index()
    client_means.rename(columns={'Conso': 'client_mean_conso'}, inplace=True)
    df_train_capped = pd.merge(df_train_capped, client_means, on='TypeClient', how='left')
    
    # 3. Ajouter la moyenne de consommation par période
    period_means = df_train.groupby(period_column)['Conso'].mean().reset_index()
    period_means.rename(columns={'Conso': 'period_mean_conso'}, inplace=True)
    df_train_capped = pd.merge(df_train_capped, period_means, on=period_column, how='left')
    
    # Encoder les types de client (one-hot encoding)
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    client_types_encoded = encoder.fit_transform(df_train_capped[['TypeClient']])
    
    # Créer un dataframe avec les types de client encodés
    client_types_df = pd.DataFrame(
        client_types_encoded,
        columns=[f"type_{client_type}" for client_type in encoder.categories_[0][1:]]
    )
    
    # AMÉLIORATION: Utiliser Ridge au lieu de LinearRegression pour la régularisation
    try:
        # Concaténer avec les autres caractéristiques
        X = pd.concat([
            df_train_capped[['period_value', 'client_mean_conso', 'period_mean_conso']], 
            client_types_df
        ], axis=1)
        
        # Utiliser la version cappée de la consommation comme cible
        y = df_train_capped['Conso_capped'].values
        
        # AMÉLIORATION: Utiliser Ridge avec régularisation pour éviter le surajustement
        alpha_value = 1.0  # Paramètre de régularisation - ajuster selon les données
        model = Ridge(alpha=alpha_value)
        
        #st.write(f"Modèle utilisé: Ridge avec alpha={alpha_value} (régularisation)")
        model.fit(X, y)

        if st.checkbox("View in-depth model diagnostics", False, key="diagnostic_ridge"):
            run_model_diagnostics_ridge(df_train)

        # Faire des prédictions pour les données d'entraînement (pour évaluer le modèle)
        y_pred = model.predict(X)
        
        # Évaluer le modèle
        metrics = evaluate_model(y, y_pred)
        
        # Ajouter l'analyse des coefficients du modèle
        #st.write("### Coefficients du modèle")
        coef_df = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': model.coef_
        })
        coef_df = coef_df.sort_values('Coefficient', ascending=False)
        #st.write("Top caractéristiques les plus influentes:")
        #st.table(coef_df.head(min(5, len(coef_df))))
        
    except Exception as e:
        st.warning(f"⚠️ Error ding model training: {str(e)}")
        st.info("Using of a simplified approach")
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
    
    # # Afficher les métriques si un modèle a été créé
    # if model is not None:
    #     st.write(f"## 📊 Performance sur les données d'entraînement")
    #     display_metrics(metrics)
        
    #     # Ajouter l'analyse des résidus
    #     st.write("## 📈 Analyse des résidus du modèle")
        
    #     try:
    #         # Utiliser la fonction plot_residuals pour afficher les graphiques des résidus
    #         plot_residuals(y, y_pred, title=f"Analyse des résidus pour la prédiction par type de client")
            
    #         # AMÉLIORATION: Ajouter un graphique de comparaison valeurs réelles vs prédites
    #         st.write("### Comparaison entre valeurs réelles et prédites")
    #         comparison_data = pd.DataFrame({
    #             'Réel': y,
    #             'Prédit': y_pred,
    #             'Erreur': y - y_pred,
    #             'Erreur (%)': (y - y_pred) / y * 100 if np.any(y != 0) else np.zeros_like(y),
    #             'TypeClient': df_train_capped['TypeClient'].values,
    #             'Période': df_train_capped[period_column].values
    #         })
            
    #         fig = go.Figure()
    #         fig.add_trace(go.Scatter(
    #             x=comparison_data['Réel'],
    #             y=comparison_data['Prédit'],
    #             mode='markers',
    #             marker=dict(color='blue'),
    #             name='Consommation'
    #         ))
            
    #         # Ajouter une ligne de référence parfaite (y=x)
    #         min_val = min(comparison_data['Réel'].min(), comparison_data['Prédit'].min())
    #         max_val = max(comparison_data['Réel'].max(), comparison_data['Prédit'].max())
    #         fig.add_trace(go.Scatter(
    #             x=[min_val, max_val],
    #             y=[min_val, max_val],
    #             mode='lines',
    #             line=dict(color='red', dash='dash'),
    #             name='Prédiction parfaite'
    #         ))
            
    #         fig.update_layout(
    #             title='Valeurs réelles vs Prédites',
    #             xaxis_title='Consommation réelle',
    #             yaxis_title='Consommation prédite',
    #             height=500
    #         )
            
    #         st.plotly_chart(fig)
            
    #     except Exception as e:
    #         st.warning(f"Impossible d'afficher les analyses supplémentaires: {str(e)}")
    
    # Prédiction pour chaque type de client pour la période suivante
    #st.write(f"## 👥 Prédiction de consommation par type de client pour {time_unit} {next_period_name} ({next_period_year})")
    
    # Obtenir la liste des types de client uniques
    client_types = df_train_capped['TypeClient'].unique()
    
    # Préparer un dataframe pour les prédictions par type de client
    client_type_predictions = []
    st.write(f"## 📈 Prediction of the consumption by client type {prediction_timeframe}")
    # Pour chaque type de client, prédire la consommation
    for client_type in client_types:
        try:
            if model is not None:
                # Créer un échantillon pour ce type de client pour la période suivante
                next_period_sample = pd.DataFrame({
                    'period_value': [next_period],
                    'client_mean_conso': [client_means[client_means['TypeClient'] == client_type]['client_mean_conso'].values[0]],
                    'period_mean_conso': [period_means[period_means[period_column] == next_period]['period_mean_conso'].values[0] 
                                        if next_period in period_means[period_column].values 
                                        else period_means['period_mean_conso'].mean()]
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
                
                # AMÉLIORATION: Limiter les prédictions extrêmes
                # Utiliser les statistiques historiques pour borner les prédictions
                client_historical = df[df['TypeClient'] == client_type]
                if not client_historical.empty:
                    client_min = client_historical['Conso'].min()
                    client_max = client_historical['Conso'].max()
                    client_mean = client_historical['Conso'].mean()
                    
                    # Définir des limites raisonnables basées sur l'historique
                    lower_limit = max(0, client_min * 0.7)  # Ne pas aller en dessous de 70% du minimum historique
                    upper_limit = client_max * 1.3  # Ne pas aller au-dessus de 130% du maximum historique
                    
                    # Limiter la prédiction
                    if pred_conso < lower_limit:
                        #st.info(f"Prédiction ajustée pour '{client_type}': {pred_conso:.2f} → {lower_limit:.2f} (limite inférieure)")
                        pred_conso = lower_limit
                    elif pred_conso > upper_limit:
                        #st.info(f"Prédiction ajustée pour '{client_type}': {pred_conso:.2f} → {upper_limit:.2f} (limite supérieure)")
                        pred_conso = upper_limit
            else:
                # Si pas de modèle, utiliser la moyenne pour ce type de client
                pred_conso = df_train_capped[df_train_capped['TypeClient'] == client_type]['Conso'].mean()
            
            # Obtenir la consommation actuelle moyenne pour ce type de client
            current_client_data = df_current[df_current['TypeClient'] == client_type]
            if current_client_data.empty:
                current_conso = df_train_capped[df_train_capped['TypeClient'] == client_type]['Conso'].mean()
            else:
                current_conso = current_client_data['Conso'].mean()
            
            # Calculer la variation avec protection contre la division par zéro
            if current_conso > 0:
                variation_pct = ((pred_conso - current_conso) / current_conso * 100)
            else:
                variation_pct = 0

            
                
            # AMÉLIORATION: Limiter les variations extrêmes pour les affichages
            max_reasonable_variation = 30.0
            if abs(variation_pct) > max_reasonable_variation:
                st.info(f"Extreme variation detected for '{client_type}': {variation_pct:.1f}%. Adjustment to ±{max_reasonable_variation}%.")
                direction = 1 if variation_pct > 0 else -1
                variation_pct = direction * max_reasonable_variation
                pred_conso = current_conso * (1 + variation_pct/100)
            
            # Ajouter au dataframe des prédictions
            client_type_predictions.append({
                'Type de Client': client_type,
                f'Current consumption': current_conso,
                f'Prediction {next_period_name} ({next_period_year})': pred_conso,
                'Variation (%)': variation_pct
            })
        except Exception as e:
            st.error(f"Error during the prediction for client {client_type}: {str(e)}")
    
    # Vérifier si nous avons des prédictions
    if not client_type_predictions:
        st.warning("⚠️ Impossible de generate prediction by client type.")
        return None, None, None
    
    # Créer un dataframe avec les prédictions
    client_type_pred_df = pd.DataFrame(client_type_predictions)
    
    # Trier par consommation prédite (décroissante)
    client_type_pred_df = client_type_pred_df.sort_values(f'Prediction {next_period_name} ({next_period_year})', ascending=False)
    
    # Afficher le tableau des prédictions
    st.table(client_type_pred_df.style.format({
        f'Current consumption': '{:.2f}',
        f'Prediction {next_period_name} ({next_period_year})': '{:.2f}',
        'Variation (%)': '{:.2f}'
    }))
    
    # Visualiser les prédictions par type de client
    fig = go.Figure()
    
    # Ajouter les valeurs actuelles
    fig.add_trace(
        go.Bar(
            x=client_type_pred_df['Type de Client'],
            y=client_type_pred_df[f'Current consumption'],
            name=f'Current consommation',
            marker_color='blue'
        )
    )
    
    # Ajouter les valeurs prédites
    fig.add_trace(
        go.Bar(
            x=client_type_pred_df['Type de Client'],
            y=client_type_pred_df[f'Prediction {next_period_name} ({next_period_year})'],
            name=f'Prediction {next_period_name} ({next_period_year})',
            marker_color='red'
        )
    )
    
    # Mettre à jour la mise en page
    fig.update_layout(
        title=f"Comparison of current and predicted consumption by client type",
        xaxis_title="Client type",
        yaxis_title="Consumption",
        barmode='group',
        height=500
    )
    
    # Afficher le graphique
    st.plotly_chart(fig)
    
       # Add detailed interpretations
    st.write("### 📝 Interpretation of results by client type")
    
    # Identify client types with the highest increases and decreases
    client_type_pred_df['Absolute_Variation'] = client_type_pred_df['Variation (%)'].abs()
    top_increasing = client_type_pred_df[client_type_pred_df['Variation (%)'] > 0].sort_values('Variation (%)', ascending=False).head(2)
    top_decreasing = client_type_pred_df[client_type_pred_df['Variation (%)'] < 0].sort_values('Variation (%)', ascending=True).head(2)
    
    # Main trends
    st.write("#### Main Trends")
    
    # Compute statistics
    avg_variation = client_type_pred_df['Variation (%)'].mean()
    num_increasing = len(client_type_pred_df[client_type_pred_df['Variation (%)'] > 0])
    num_decreasing = len(client_type_pred_df[client_type_pred_df['Variation (%)'] < 0])
    num_stable = len(client_type_pred_df[(client_type_pred_df['Variation (%)'] >= -1) & (client_type_pred_df['Variation (%)'] <= 1)])
    
    # Display global trend
    if avg_variation > 3:
        st.write(f"🔺 **Overall increasing trend**: On average, client types are expected to experience a {avg_variation:.1f}% increase in water consumption.")
    elif avg_variation < -3:
        st.write(f"🔻 **Overall decreasing trend**: On average, client types are expected to experience a {abs(avg_variation):.1f}% decrease in water consumption.")
    else:
        st.write(f"↔️ **Stable overall trend**: On average, client types are expected to experience a limited variation of {avg_variation:.1f}% in water consumption.")
    
    st.write(f"- {num_increasing} client types show an increasing trend")
    st.write(f"- {num_decreasing} client types show a decreasing trend")
    st.write(f"- {num_stable} client types have relatively stable consumption (variation between -1% and +1%)")
    
    # Key client types to monitor
    st.write("#### Key Client Types to Monitor")
    
    if not top_increasing.empty:
        st.write("**Client types with the highest predicted increase:**")
        for i, row in top_increasing.iterrows():
            st.write(f"🔺 **{row['Type de Client']}**: +{row['Variation (%)']:.1f}% (from {row[f'Current consumption']:.1f} to {row[f'Prediction {next_period_name} ({next_period_year})']:.1f})")
    
    if not top_decreasing.empty:
        st.write("**Client types with the highest predicted decrease:**")
        for i, row in top_decreasing.iterrows():
            st.write(f"🔻 **{row['Type de Client']}**: {row['Variation (%)']:.1f}% (from {row[f'Current consumption']:.1f} to {row[f'Prediction {next_period_name} ({next_period_year})']:.1f})")
    
    # Potential explanatory factors
    st.write("#### Potential Explanatory Factors")
    st.write("Variations in consumption among client types can be explained by several factors:")
    st.write("- **Economic factors**: changes in business activity, seasonal demand variations")
    st.write("- **Behavioral factors**: changes in consumption habits")
    st.write("- **Structural factors**: evolution in the number of clients within each category")
    st.write("- **Regulatory factors**: new regulations or pricing affecting certain categories")
    
    # Business implications
    st.write("#### Business Implications")
    st.write("These predictions can have the following implications for business strategy:")
    
    # Identify the client type with the highest predicted consumption
    if len(client_type_pred_df) > 0:
        top_consumer = client_type_pred_df.iloc[0]['Type de Client']
        st.write(f"- Clients of type **{top_consumer}** represent the segment with the highest predicted consumption")
    
    # Recommendations based on variations
    if num_increasing > num_decreasing:
        st.write("- The **general increase** in consumption suggests an opportunity to optimize services and pricing")
    elif num_decreasing > num_increasing:
        st.write("- The **general decrease** in consumption suggests a need for customer retention and incentive programs")
    
    # Recommendations
    st.write("#### Recommendations")
    st.write("Based on these predictions, we recommend:")
    st.write("1. **Adapting the service offering** according to trends by client type")
    
    if not top_increasing.empty:
        st.write(f"2. **Preparing resources** to meet the increased demand from clients of type {', '.join(top_increasing['Type de Client'].head(1).tolist())}")
    
    if not top_decreasing.empty:
        st.write(f"3. **Developing retention programs** for declining segments such as {', '.join(top_decreasing['Type de Client'].head(1).tolist())}")
    
    st.write("4. **Optimizing pricing** based on demand elasticity by segment")
    st.write("5. **Implementing commercial monitoring** for client types with significant variations")
    
    # Factors influencing prediction reliability
    st.write("**Factors Influencing Prediction Reliability:**")
    st.write("- Available data quantity: " + ("✅ Sufficient" if len(df_train) > 50 else "⚠️ Limited"))
    st.write("- Diversity of client types: " + ("✅ Good" if len(client_types) > 3 else "⚠️ Limited"))
    st.write("- Regularity of temporal data: " + ("✅ Regular" if len(df_train) > len(client_types) * 3 else "⚠️ Irregular"))
    st.write("- Presence of extreme values: " + ("⚠️ Significant" if outliers_count/len(df_train)*100 > 10 else "✅ Moderate"))

    # # AMÉLIORATION: Visualisation de la distribution des prédictions
    # st.write("#### Distribution des prédictions")
    
    # try:
    #     # Créer des histogrammes pour comparer les distributions actuelles et prédites
    #     hist_data = [
    #         client_type_pred_df[f'Consommation {current_period_name} ({current_year})'],
    #         client_type_pred_df[f'Prédiction {next_period_name} ({next_period_year})']
    #     ]
        
    #     fig = go.Figure()
    #     fig.add_trace(go.Histogram(
    #         x=hist_data[0],
    #         name=f'Consommation {current_period_name}',
    #         opacity=0.7,
    #         marker=dict(color='blue')
    #     ))
        
    #     fig.add_trace(go.Histogram(
    #         x=hist_data[1],
    #         name=f'Prédiction {next_period_name}',
    #         opacity=0.7,
    #         marker=dict(color='red')
    #     ))
        
    #     fig.update_layout(
    #         title="Distribution des consommations actuelles vs prédites",
    #         xaxis_title='Consommation',
    #         yaxis_title='Fréquence',
    #         barmode='overlay',
    #         height=400
    #     )
        
    #     st.plotly_chart(fig)
        
    # except Exception as e:
    #     st.warning(f"Impossible d'afficher la distribution des prédictions: {str(e)}")

    return model, client_types, client_type_pred_df
# Mettre à jour la fonction dashboard pour inclure les nouvelles prédictions
def prediction_dashboard(df_filtered, selected_user, selected_year, prediction_timeframe, is_synthetic=False):
    """
    Cette fonction intègre les modèles de prédiction dans l'interface principale,
    prenant en compte les modifications apportées aux fonctions de prédiction.
    
    Args:
        df_filtered: DataFrame filtré selon les critères de l'utilisateur
        selected_user: Utilisateur sélectionné
        selected_year: year sélectionnée
        prediction_timeframe: Période à prédire (Mois prochain, Trimestre prochain, etc.)
    """
    # Titre de la section
    #st.write("Cette analyse utilise des modèles de régression linéaire pour prédire la consommation d'eau pour la période future.")
    
    # Afficher l'information sur l'utilisateur et l'year sélectionnée
    if selected_user != "All users":
        st.write(f"**👤 User:** {selected_user}")
    
    # Afficher l'year sélectionnée
    st.write(f"**📅 Reference year:** {selected_year}")

    if is_synthetic:
        st.info("⚠️ Using synthetic data for predictions. Results are for demonstration only.")
    
    
    # Vérifier si nous avons des données disponibles
    if df_filtered.empty:
        st.warning("⚠️ No data available with current filters. Try widening your filters.")
        return
    
    # Utiliser les mêmes onglets que dans le dashboard principal
    tab1, tab2, tab3 = st.tabs(["📅 by period", "🏙️ by city", "👥 By Client type"])
    
    with tab1:
        # Prédiction par période - utilise directement le prediction_timeframe
        predict_by_period(df_filtered, prediction_timeframe, prediction_timeframe, selected_year)
    
    with tab2:
        # Prédiction par ville - utilise directement le prediction_timeframe
        predict_by_city(df_filtered, prediction_timeframe, selected_year)
    
    with tab3:
        # Prédiction par type de client - utilise directement le prediction_timeframe
        predict_by_client_type(df_filtered, prediction_timeframe, selected_year)
