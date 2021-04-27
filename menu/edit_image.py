import streamlit as st
from menu.edit.color import saturation, brightness
from utils.utils import load_image

def edit_image():
    image_file = st.file_uploader("이미지 파일 업로드", type = ['png', 'jpeg', 'jpg'], accept_multiple_files=False)

    if image_file is not None:

        img = load_image(image_file)

        menu_list = ['Saturation','Brightness','hello']
        selected_menu_list = st.multiselect("언어를 선택하세요", menu_list)


        if selected_menu_list is None:
            return st.write('"err_code" : 1')
        if 'Saturation' in selected_menu_list:
            image = saturation(img)
            st.image(image)
        if 'Brightness' in selected_menu_list:
            image = brightness(img)
            st.image(image)
            