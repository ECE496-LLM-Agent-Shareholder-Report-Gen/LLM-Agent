import streamlit as st
from GUI.session_page import SessionPage

# Sidebar
with st.sidebar:
    st.title('Sidebar')
    'Session 1'
    'Session 2'
    'Session 3'

# Main Content
session = SessionPage()
session.render()
