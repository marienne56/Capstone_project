import streamlit as st
from dbConnection import get_connection
import pandas as pd

def get_user_role(identifier):
    """
    Récupère le rôle d'un utilisateur basé sur son identifiant
    en utilisant la colonne role_id (où 123 = admin)
    """
    try:
        engine = get_connection()
        query = """
        SELECT role_id FROM users 
        WHERE identifier = %s
        """
        df = pd.read_sql(query, engine, params=(identifier,))
        if not df.empty and df['role_id'].iloc[0] == 123:
            return "admin"
        return "user"  # Rôle par défaut pour tous les autres
    except Exception as e:
        st.error(f"Erreur lors de la récupération du rôle: {e}")
        return "user"  # En cas d'erreur, considérer comme utilisateur standard

def check_admin_permission(identifier):
    """Vérifie si un utilisateur a les permissions admin"""
    role = get_user_role(identifier)
    return role.lower() == "admin"