from dbConnection import get_connection
from sqlalchemy import text
import streamlit as st


def delete_user(identifier):

    if "delete_confirmation" not in st.session_state:
     st.session_state["delete_confirmation"] = None
    try:
        engine = get_connection()
        with engine.connect() as conn:
            query = text('DELETE FROM users WHERE identifier = :identifier')
            conn.execute(query, {"identifier": identifier})
            conn.commit()  # Appliquer les changements
        st.success(f"The user withidentifier {identifier} has been successfully deleted.")
    except Exception as e:
        st.error(f"Error deleting user : {str(e)}")

if __name__ == "__main__":
    identifier = st.session_state.get("identifier", None)
    if identifier:
        delete_user(identifier)
    else:
        st.error("No identifier found. Please log in first.")