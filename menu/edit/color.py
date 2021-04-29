import streamlit as st
import cv2
import numpy as np

#컬러에 관한 함수는 여기다 만들자.

def saturation(image):

    image_np = np.array(image)

    #이미지가 아무것도 없다면.
    if image_np is None:
        print("Could not open or find the image")

    saturationScale = st.sidebar.slider('채도', min_value = 0.00, max_value = 10.00, value = 1.00)

    hsvImage = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)

    hsvImage = np.float32(hsvImage)

    H, S, V = cv2.split(hsvImage)

    S = np.clip(S * saturationScale, 0 ,255)

    hsvImage = cv2.merge([H,S,V])

    hsvImage = np.uint8(hsvImage)

    imgBgr=cv2.cvtColor(hsvImage, cv2.COLOR_HSV2BGR)

    return imgBgr


def brightness(image):

    image_np = np.array(image)

    #이미지가 아무것도 없다면.
    if image_np is None:
        print("Could not open or find the image")

    beta = st.sidebar.slider('밝기', min_value = -300, max_value = 300, value = 0)

    #컬러스페이스는 ycrcd로 정했다. rgb hsv 같은 것.
    #불러온 이미지의 컬러 스페이스를 ycrcb로 변경했다.
    ycbImage = cv2.cvtColor(image_np, cv2.COLOR_BGR2YCrCb)



    #가공을 위해서 uint8을 float으로 바꾼다. 오버플로우 막기 위해.
    ycbImage = np.float32(ycbImage)


    #채널을 분리한다.
    Ychannel, Cr, Cb = cv2.split(ycbImage)

    #밝기 조절. 이 바운더리 안에서 밝기를 조절해라.
    Ychannel = np.clip(Ychannel + beta, 0, 255)

    #다시 정수로 바꿔서 합쳐줘야한다.
    #합치고 정수로 바꿔도 상관없다.
    #합칠 땐 리스트 형식으로 준다.
    ycbImage = cv2.merge([Ychannel, Cr, Cb])

    #다시 uint8로 변경(정수)
    ycbImage = np.uint8(ycbImage)


    #화면 표시를 위해서는 컬러 스페이스 BGR 로 변경

    ycbImage = cv2.cvtColor(ycbImage, cv2.COLOR_YCrCb2BGR)

    return ycbImage


def colortransfer():
    pass