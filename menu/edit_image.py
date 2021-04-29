import streamlit as st
from menu.edit.color import saturation, brightness
from utils.utils import load_image

def edit_image():
    menu_list = ['Select menu','Saturation','Brightness','hello']
    selected_menu_list = st.sidebar.selectbox("메뉴를 고르세요.", menu_list)
    # .multiselect("언어를 선택하세요", menu_list)
    image_file = st.sidebar.file_uploader("이미지 파일 업로드", type = ['png', 'jpeg', 'jpg'], accept_multiple_files=False)

    if image_file is not None:

        img = load_image(image_file)


        if selected_menu_list is None:
            return st.write('"err_code" : 1')
        if 'Saturation' in selected_menu_list:
            image = saturation(img)
            st.image(image)
        elif 'Brightness' in selected_menu_list:
            image = brightness(img)
            st.image(image)
        else:
            st.write("메뉴를 고르세요.")
            