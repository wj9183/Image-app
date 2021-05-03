import streamlit as st
from menu.edit.color import saturation, brightness, colortransfer, median, gaussian, convolution, cooling, gamma, contrastscailing
from utils.utils import load_image

def edit_image():
    menu = ['Select menu','Saturation','Brightness','Color transfer', 'Median', 'Gaussian', 'Convolution', 'Cooling','Gamma','Contrast Scailing']
    selected_menu= st.sidebar.selectbox("메뉴를 고르세요.", menu)
    # .multiselect("언어를 선택하세요", menu_list)
    image_file = st.sidebar.file_uploader("이미지 파일 업로드", type = ['png', 'jpeg', 'jpg'], accept_multiple_files=False)

    if image_file is not None:

        img = load_image(image_file)


        if selected_menu is None:
            return st.write('"err_code" : 1')
        if selected_menu == 'Saturation':
            image = saturation(img)
            st.image(image)
        elif selected_menu == 'Brightness':
            image = brightness(img)
            st.image(image)
        elif selected_menu == 'Color transfer':
            image_file2 = st.sidebar.file_uploader("합성할 파일 업로드", type = ['png', 'jpeg', 'jpg'], accept_multiple_files=False)
            
            if image_file2 is not None:

                img2 = load_image(image_file2)
                image = colortransfer(img,img2)
                st.image(image)
        elif selected_menu == 'Median':
            image = median(img)
            st.image(image)
        elif selected_menu == 'Gaussian':
            image = gaussian(img)
            st.image(image)
        elif selected_menu == 'Convolution':
            image = convolution(img)
            st.image(image)
        elif selected_menu == 'Cooling':
            image = cooling(img)
            st.image(image)
        elif selected_menu == 'Gamma':
            image = gamma(img)
            st.image(image)
        elif selected_menu =='Contrast Scailing':
            image = contrastscailing(img)
            st.image(image)
        else:
            st.write("메뉴를 고르세요.")
            