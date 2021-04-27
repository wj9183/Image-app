import streamlit as st
import cv2
from menu.about_app import about_app
from menu.edit_image import edit_image

def main():
    menu = ['About app', 'Edit image', 'Example']
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == 'About app':
        about_app()
    elif choice == 'Edit image':
        edit_image()
    elif choice == 0:
        pass
    






if __name__ == '__main__':
        main()
