import streamlit as st
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
from datetime import datetime
from dbConnection import get_connection

# Database functions for threshold management

def get_user_billing_threshold(identifier):
    """Retrieve the billing threshold for a specific user."""
    try:
        engine = get_connection()
        query = """
        SELECT threshold_value FROM billing_thresholds 
        WHERE identifier = %s
        """
        # Use tuple for params
        df = pd.read_sql(query, engine, params=(identifier,))
        if not df.empty:
            return df['threshold_value'].iloc[0]
        return None
    except Exception as e:
        st.error(f"Error fetching user billing threshold: {e}")
        return None

def set_billing_threshold(identifier, threshold_value):
    """Set or update a billing threshold for a user."""
    try:
        engine = get_connection()
        with engine.connect() as conn:
            # Check if threshold already exists
            from sqlalchemy import text
            
            check_query = text("""
            SELECT id FROM billing_thresholds 
            WHERE identifier = :identifier
            """)
            result = conn.execute(check_query, {"identifier": identifier})
            existing_id = result.fetchone()
            
            if existing_id:
                # Update existing threshold
                update_query = text("""
                UPDATE billing_thresholds
                SET threshold_value = :threshold_value, updated_at = CURRENT_TIMESTAMP
                WHERE id = :id
                """)
                conn.execute(update_query, {
                    "threshold_value": threshold_value, 
                    "id": existing_id[0]
                })
                conn.commit()  # Ajout explicite du commit
                return f"Billing threshold updated to {threshold_value} FCFA"
            else:
                # Insert new threshold
                insert_query = text("""
                INSERT INTO billing_thresholds (identifier, threshold_value)
                VALUES (:identifier, :threshold_value)
                """)
                conn.execute(insert_query, {
                    "identifier": identifier, 
                    "threshold_value": threshold_value
                })
                conn.commit()  # Ajout explicite du commit
                return f"New billing threshold set to {threshold_value} FCFA"
    except Exception as e:
        st.error(f"Error setting billing threshold: {e}")
        import traceback
        st.error(traceback.format_exc())
        return f"Failed to set threshold: {str(e)}"
# def reset_billing_threshold(identifier):
#     """Reset (delete) the billing threshold for a user."""
#     try:
#         engine = get_connection()
#         with engine.connect() as conn:
#             delete_query = "DELETE FROM billing_thresholds WHERE identifier = %s"
#             conn.execute(delete_query, (identifier,))
#         return "Billing threshold reset successfully"
#     except Exception as e:
#         st.error(f"Error resetting billing threshold: {e}")
#         return f"Failed to reset threshold: {str(e)}"
@st.cache_data(ttl=600)
def get_user_email(identifier):
    """Get the email address for a user."""
    try:
        engine = get_connection()
        query = "SELECT email FROM users WHERE identifier = %s"
        df = pd.read_sql(query, engine, params=(identifier,))
        if not df.empty and df['email'].iloc[0]:
            return df['email'].iloc[0]
        return None
    except Exception as e:
        st.error(f"Error fetching user email: {e}")
        return None
@st.cache_data(ttl=600)
def get_user_id_from_identifier(identifier):
    """Get the user_id for a given identifier."""
    try:
        engine = get_connection()
        query = "SELECT user_id FROM users WHERE identifier = %s"
        df = pd.read_sql(query, engine, params=(identifier,))
        if not df.empty:
            return df['user_id'].iloc[0]
        return None
    except Exception as e:
        st.error(f"Error fetching user ID: {e}")
        return None

def send_billing_alert_email(to_email, subject, message):
    """Send an email alert using Mailjet."""
    try:
        import requests
        import json
        import socket
        
        # Mailjet credentials
        api_key = "9db4d134407c233f94673231a25ba13b"
        api_secret = "01e7c939ad6d27232be24fa641e48e23"
        
        url = "https://api.mailjet.com/v3.1/send"
        
        # Check internet connectivity before sending
        try:
            socket.create_connection(("www.google.com", 80))
        except (socket.error, socket.timeout):
            st.error("❌ No internet connection. Please check your network settings.")
            return False
        
        # Prepare email data
        data = {
            'Messages': [
                {
                    "From": {
                        "Email": "mariennedosso@gmail.com",  # Must be a verified email in your Mailjet account
                        "Name": "Water Consumption Alert"
                    },
                    "To": [
                        {
                            "Email": to_email,
                            "Name": "User"
                        }
                    ],
                    "Subject": subject,
                    "HTMLPart": message
                }
            ]
        }
        
        # Send request to Mailjet API
        response = requests.post(
            url,
            auth=(api_key, api_secret),
            json=data
        )
        
        # Check response
        if response.status_code == 200:
            return True
        else:
            st.error("❌ Failed to send email. Network or server error.")
            return False
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Unable to connect to email server. Check your internet connection.")
        return False
    except Exception as e:
        st.error(f"❌ Unexpected error sending email: {e}")
        return False
def create_alert(user_identifier, alert_type, message, prediction_id=None):
    """Create an alert in the alert table."""
    try:
        # Get user_id from identifier
        user_id = get_user_id_from_identifier(user_identifier)
        if not user_id:
            #st.error(f"Could not find user_id for identifier: {user_identifier}")
            return False
        
        # Debug messages
        # st.info(f"DEBUG: Creating alert for user ID: {user_id}")
        # st.info(f"DEBUG: Alert type: {alert_type}")
        
        # Import text from sqlalchemy
        from sqlalchemy import text
        
        engine = get_connection()
        with engine.connect() as conn:
            # Créer un dictionnaire pour les paramètres
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            if prediction_id is not None:
                insert_query = text("""
                INSERT INTO alert 
                (alert_type, created_at, message, user_id, prediction_id) 
                VALUES (:alert_type, :created_at, :message, :user_id, :prediction_id)
                """)
                params = {
                    "alert_type": alert_type,
                    "created_at": current_date,
                    "message": message,
                    "user_id": user_id,
                    "prediction_id": prediction_id
                }
            else:
                # Sans prediction_id
                insert_query = text("""
                INSERT INTO alert 
                (alert_type, created_at, message, user_id) 
                VALUES (:alert_type, :created_at, :message, :user_id)
                """)
                params = {
                    "alert_type": alert_type,
                    "created_at": current_date,
                    "message": message,
                    "user_id": user_id
                }
            
            # Exécuter avec le dictionnaire de paramètres
            conn.execute(insert_query, params)
            conn.commit()  # Assurez-vous de commit les changements
            
            #st.success(f"DEBUG: Alert created successfully!")
            return True
    except Exception as e:
        st.error(f"Error creating alert: {e}")
        import traceback
        st.error(traceback.format_exc())
        return False
def check_billing_threshold(identifier, predicted_amount, current_amount, period_name):
    """Check if current billing amount exceeds threshold and prepare notification."""
    try:
        # Logging minimal
        # st.write(f"- identifier: {identifier}")
        # st.write(f"- predicted_amount: {predicted_amount:.2f}")
        # st.write(f"- current_amount: {current_amount:.2f}")
        # st.write(f"- period_name: {period_name}")
        
        if identifier == "All users":
            return None
            
        # Get user threshold
        threshold = get_user_billing_threshold(identifier)
        
        # CAS 1: Vérifier si un seuil est défini et si le montant actuel le dépasse
        if threshold is not None:
            if current_amount >= threshold:
                # Create notification data for threshold exceeded
                percent_over = ((current_amount - threshold) / threshold * 100) if threshold > 0 else 0
                notification = {
                    'type': 'billing_threshold',
                    'message': f"🚨 Current billing of {current_amount:.2f} FCFA exceeds your threshold of {threshold:.2f} FCFA!",
                    'predicted': predicted_amount,
                    'current': current_amount,
                    'threshold': threshold,
                    'percent_over': percent_over
                }
                
                # Create alert in the database
                try:
                    alert_created = create_alert(
                        identifier, 
                        'billing_threshold', 
                        f"Current billing of {current_amount:.2f} FCFA for {period_name} exceeds your threshold of {threshold:.2f} FCFA."
                    )
                except Exception as alert_err:
                    st.warning(f"Warning: Error creating threshold alert: {alert_err}")
                
                return notification
            else:
                # st.write("Threshold not exceeded")
                pass
        # CAS 2: Si aucun seuil n'est défini, vérifier si le montant prédit est supérieur au montant actuel
        else:
            # st.write("No threshold set, checking for billing increase")
            
            if predicted_amount >= current_amount:
                # Calcule le pourcentage d'augmentation
                percent_increase = ((predicted_amount - current_amount) / current_amount * 100) if current_amount > 0 else 0
                
                # Alerte si l'augmentation est supérieure à 0%
                if percent_increase >= 0:
                    notification = {
                        'type': 'billing_increase',
                        'message': f"⚠️ Predicted billing for {period_name} is {percent_increase:.1f}% higher than your current billing.",
                        'predicted': predicted_amount,
                        'current': current_amount,
                        'threshold': None,
                        'percent_increase': percent_increase
                    }
                    
                    # Créer une alerte en base de données
                    try:
                        alert_created = create_alert(
                            identifier,
                            'billing_increase',
                            f"Predicted billing of {predicted_amount:.2f} FCFA for {period_name} is {percent_increase:.1f}% higher than your current billing of {current_amount:.2f} FCFA."
                        )
                    except Exception as alert_err:
                        st.warning(f"Warning: Error creating billing increase alert: {alert_err}")
                    
                    return notification
                else:
                    # st.write("No significant increase detected")
                    pass
            else:
                # st.write("Predicted amount is not higher than current amount")
                pass
            
        return None
    except Exception as e:
        st.error(f"Error checking billing threshold: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None
def get_user_alerts(identifier, limit=5):
    """
    Get recent alerts for a user.
    
    Args:
        identifier: User identifier
        limit: Maximum number of alerts to return
        
    Returns:
        DataFrame: Recent alerts
    """
    try:
        # Get user_id from identifier
        user_id = get_user_id_from_identifier(identifier)
        if not user_id:
            return pd.DataFrame()
        
        engine = get_connection()
        query = """
        SELECT alert_id, alert_type, created_at, message
        FROM alert
        WHERE user_id = %s AND (alert_type = 'billing_threshold' OR alert_type = 'billing_increase')
        ORDER BY created_at DESC
        LIMIT %s
        """
        
        # Use tuple for params
        alerts_df = pd.read_sql(query, engine, params=(user_id, limit))
        return alerts_df
    except Exception as e:
        st.error(f"Error fetching user alerts: {e}")
        return pd.DataFrame()
    

    # Ajoutez ce code après l'appel à get_user_alerts pour déboguer
    alerts_df = get_user_alerts(selected_user)
    st.write(f"DEBUG: Retrieved {len(alerts_df)} alerts from database")
    if alerts_df.empty:
        st.write("DEBUG: Alert dataframe is empty. Checking possible reasons...")
        
        # Vérifier si l'utilisateur existe
        user_id = get_user_id_from_identifier(selected_user)
        st.write(f"DEBUG: User ID for {selected_user} is: {user_id}")
        
        # Vérifier s'il y a des alertes dans la table sans filtres
        engine = get_connection()
        query = "SELECT COUNT(*) as count FROM alert"
        count_df = pd.read_sql(query, engine)
        st.write(f"DEBUG: Total alerts in database: {count_df['count'].iloc[0]}")

def display_threshold_management(selected_user):
    """Display billing threshold management UI."""
    st.write("## 💰 Billing Threshold Management")
    
    if selected_user == "All users":
        st.warning("Please select a specific user to manage billing thresholds.")
        return
    
    # Get current threshold
    current_threshold = get_user_billing_threshold(selected_user)
    
    # Display current threshold
    if current_threshold is not None:
        st.success(f"Current billing threshold: **{current_threshold:.2f} FCFA**")
    else:
        st.info("No billing threshold currently set.")
    
    # Form to set threshold
    with st.form("threshold_form"):
        new_threshold = st.number_input(
            "Set Billing Threshold (FCFA)", 
            min_value=0.0, 
            value=current_threshold if current_threshold is not None else 5000.0,
            step=100.0
        )
        
        col1, col2 = st.columns(2)
        with col1:
            submit = st.form_submit_button("Save Threshold")
       
        
        if submit:
            #result = 
            result =set_billing_threshold(selected_user, new_threshold)
            
            st.success(result)
            
            st.rerun()
        