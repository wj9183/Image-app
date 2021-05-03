import streamlit as st
import cv2
import numpy as np




def saturation(image):

    image_np = np.array(image)

    #이미지가 아무것도 없다면.
    if image_np is None:
        print("Could not open or find the image")

    saturationScale = st.sidebar.slider('Saturation', min_value = 0.00, max_value = 10.00, value = 1.00)

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

    beta = st.sidebar.slider('Brightness', min_value = -300, max_value = 300, value = 0)

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


def colortransfer(image,image2):

    src = np.array(image)
    dst = np.array(image2)

    cv2.imshow("src", src)
    cv2.imshow("dst", dst)

    output = dst.copy()


    #내가 알기로는 오픈씨브이가 BGR을 사용해서 바꿔주는 것 같다.
    srcLab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    dstLab = cv2.cvtColor(dst, cv2.COLOR_BGR2LAB)
    outputLab = cv2.cvtColor(output, cv2.COLOR_BGR2LAB)

    #float으로 바꿔놓고 연산작업을 해라.,
    srcLab = srcLab.astype('float')
    dstLab = dstLab.astype('float')
    outputLab = outputLab.astype('float')

    #채널 분리
    srcL, srcA, srcB = cv2.split(srcLab)
    dstL, dstA, dstB = cv2.split(dstLab)
    outL, outA, outB = cv2.split(outputLab)

    outL = dstL - dstL.mean()
    outA = dstA - dstA.mean()
    outB = dstB - dstB.mean()

    #우리가 얻고자하는 이미지
    outL = outL * (srcL.std() / dstL.std())
    outA = outA * (srcA.std() / dstA.std())
    outB = outB * (srcB.std() / dstB.std())

    outL = outL + srcL.mean()
    outA = outA + srcA.mean()
    outB = outB + srcB.mean()

    #우리가 눈으로 보기 위해서는? 사진은 0~255 사이 값으로 세팅.
    outL = np.clip(outL, 0, 255)
    outA = np.clip(outA, 0, 255)
    outB = np.clip(outB, 0, 255)
    # 채널 합치기
    outputLab = cv2.merge( [outL, outA, outB] )
    #이미지는 8비트(1바이트) 정수이므로, 형변환을 해준다.
    outputLab = np.uint8(outputLab)

    #imshow는 BGR이므로, (opencv는 비지알로 처리하기 때문에)
    outputLab = cv2.cvtColor(outputLab, cv2.COLOR_LAB2BGR)

    return outputLab


def median(image):


    img = np.array(image)

    median_ksize = st.sidebar.slider('Kernel size', min_value = 1, max_value = 30, value = 5)

    # 5x5 커널 사용할 때 
    dst = cv2.medianBlur(img, ksize = median_ksize)

    #표준 편차가 클수록 차트는 종모양처럼 된다.

    return dst


def gaussian(image):
    img = np.array(image)

    #우리가 이전처럼 복잡하게 할 필요 없이,
    #opencv에서 이미 만들어서 넣어놨다.
    gaussian_ksize = st.sidebar.slider('Kernel size', min_value = 1, max_value = 30, value = 5)
    sigmaX = st.sidebar.slider('SigmaX', min_value = 1, max_value = 30, value = 5)
    
    #이거 하나 하면 컨볼루션까지 다 한 이미지가 변수에 저장됨.
    # 3x3 커널 사용할 때 
    dst = cv2.GaussianBlur(img, (gaussian_ksize, gaussian_ksize), sigmaX = sigmaX)

    #표준 편차가 클수록 차트는 종모양처럼 된다.

    return dst

def boxfilter(image):
    img = np.array(image)

    #우리가 이전처럼 복잡하게 할 필요 없이,
    #opencv에서 이미 만들어서 넣어놨다.
    boxfilter_ksize = st.sidebar.slider('Kernel size', min_value = 1, max_value = 30, value = 5)
    #이거 하나 하면 컨볼루션까지 다 한 이미지가 변수에 저장됨.
    # 3x3 커널 사용할 때 
    dst = cv2.blur(img, (boxfilter_ksize, boxfilter_ksize))

    return dst


def convolution(image):
    img = np.array(image)
    # cv2.imshow("original", img)
    convolution_ksize = st.sidebar.slider('Kernel size', min_value = 1, max_value = 30, value = 5)
    #이 이미지를 컨볼루션을 통해 필터링하겠다.
    #커널은 행과 열이 동일해야한다.
    kernel = np.ones((convolution_ksize , convolution_ksize )) / convolution_ksize **2
    #이러면 255 웬만하면 안넘을 것임.

 

    # 컨볼루션! cv2.filter2D 함수
    result = cv2.filter2D(img, -1, kernel)

    return result


#얘를 어떻게 조절할 수 있게 만들까.
def cooling(image):
    original = np.array(image)

    img = original.copy()



    #커브를 그릴 거임.
    #x축 피봇 포인트
    originalValue = np.array([0,50,100,150,200,255]) #이게 x축

    # y축 포인트 : 빨간쪽과 파란쪽, 두 부분의 포인트.
    bCurve = np.array([0,80,150,190,220,255])

    rCurve = np.array([0,20,40,75,150,255])

    #머릿속에서 r커브 b커브 두개를 차트로 그려봐라.
    #포토샵 색보정 할때 했던 그거다. 물론 정확히 그거는 아니겠지만.

    #Lookup 테이블 만들기
    fullrange = np.arange(0,255+1)
    rLUT = np.interp(fullrange, originalValue, rCurve)
    bLUT = np.interp(fullrange, originalValue, bCurve)

    print(rLUT)
    print(rLUT.shape) #0부터 255까지 했기 때문에 256이 나온다.

    #이미지는 지금 컬러스페이스가 bgr이고, 0,1,2일 것임.
    #r채널이랑 행렬만 가져옴. 아까 split해서 r채널만 갖고온 거랑 똑같다.
    # B, G, rChannel = cv2.split(img)
    rChannel = img[ : , :, 2]
    rChannel = cv2.LUT(rChannel, rLUT)
    img[ : , : , 2] = rChannel

    #블루 채널도 적용해주자
    bChannel = img[:,:,0]
    bChannel = cv2.LUT(bChannel, bLUT)

    img[ : , :, 0] = bChannel

    return img


def gamma(image):
    img = np.array(image)
    gamma = st.sidebar.slider('Gamma', min_value = 0.0, max_value = 20.0, value = 1.0)

    #이러면 0부터 255다. 헷갈리지 말기.
    fullRange = np.arange(0,256)


    lookupTable = np.uint8(255 * np.power( (fullRange / 255.0) , gamma ))

    #LUT는 룩업테이블 약자로 쓴 것
    output = cv2.LUT(img, lookupTable)

    return output

def contrastscailing(image):
    
    img = np.array(image)

    scaleFactor = st.sidebar.slider('ScaleFactor', min_value = 0, max_value = 200, value = 0)

    ycbImage = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    ycbImage = ycbImage.astype(float)

    Ychannel, Cr, Cb = cv2.split(ycbImage)

    Ychannel = np.clip(Ychannel * scaleFactor, 0 ,255)

    ycbImage = cv2.merge([Ychannel, Cr, Cb])

    ycbImage = np.uint8(ycbImage)

    ycbImage = cv2.cvtColor(ycbImage, cv2.COLOR_YCrCb2BGR)

    return ycbImage