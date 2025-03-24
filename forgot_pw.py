import streamlit as st
import hashlib
from sqlalchemy import text
from dbConnection import get_connection

def hash_password(password):
    """Hash password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def password_strength_check(password):
    """Check if password meets minimum requirements."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    
    has_uppercase = any(c.isupper() for c in password)
    has_lowercase = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(not c.isalnum() for c in password)
    
    if not (has_uppercase and has_lowercase and has_digit and has_special):
        return False, "Password must contain uppercase, lowercase, digit, and special character."
    
    return True, "Password meets requirements."

st.markdown("""
    <style>
        .stTextInput > label {font-weight: bold;}
        .stSelectbox > label {font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

def password_change():
    if "page" not in st.session_state:
        st.session_state.page = "forgot_password" 
    
    st.title("Reset Your :violet[Password]")
    
    with st.form("password_reset_form", clear_on_submit=True):
        identifier = st.text_input("Enter your identifier").strip()
        new_password = st.text_input("New password", type="password")
        confirm_password = st.text_input("Confirm new password", type="password")
        submit_button = st.form_submit_button("Reset Password")
    
    if submit_button:
        if not identifier:
            st.error("Please enter your identifier.")
            return
            
        if not new_password or not confirm_password:
            st.error("Please enter and confirm your new password.")
            return
            
        if new_password != confirm_password:
            st.error("Passwords do not match. Please try again.")
            return
            
        # Check password strength
        valid_password, message = password_strength_check(new_password)
        if not valid_password:
            st.error(message)
            return
            
        try:
            engine = get_connection()
            with engine.connect() as conn:
                # First, check if the identifier exists
                check_query = text("""
                SELECT identifier FROM users
                WHERE identifier = :identifier
                """)
                
                user_exists = conn.execute(check_query, {"identifier": identifier}).fetchone()
                
                if not user_exists:
                    st.error("This identifier is not linked to any account.")
                    return
                    
                # Update password
                update_query = text("""
                UPDATE users
                SET password_hash = :password_hash
                WHERE identifier = :identifier
                """)
                
                conn.execute(update_query, {
                    "identifier": identifier,
                    "password_hash": hash_password(new_password)
                })
                
                conn.commit()
                
                st.success("Password has been changed successfully!")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please try again or contact support.")
    
    # Bouton pour aller à la page de connexion (toujours affiché)
    if st.button("Go to Login Page"):
        st.session_state.page = "login"
        st.rerun()
if __name__ == "__main__":
    password_change()
