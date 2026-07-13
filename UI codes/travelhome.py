# main.py

import streamlit as st
from catnlp import main
from page2 import display_page2
from page3 import display_page3

def main():
    st.sidebar.title('Navigation')
    page_options = {
        'Page 1': display_page1,
        'Page 2': display_page2,
        'Page 3': display_page3
    }
    selected_page = st.sidebar.radio('Go to', list(page_options.keys()))
    page_options[selected_page]()

if __name__ == "__main__":
    main()
