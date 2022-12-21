
import streamlit as st
import cv2
import numpy as np
#import torch
import base64
#import matplotlib.pyplot as plt
#import tensorflow
from tensorflow_addons.layers import InstanceNormalization
from keras.utils import load_img
from keras.utils import array_to_img
from keras.utils import img_to_array
from keras.models import load_model
from streamlit_option_menu import option_menu
from urllib.request import urlopen
from io import BytesIO

def get_image_download_link(img):
	"""Generates a link allowing the PIL image to be downloaded
	in:  PIL image
	out: href string
	"""
	buffered = BytesIO()
	img.save(buffered, format="JPEG")
	img_str = base64.b64encode(buffered.getvalue()).decode()
	href = f'<a download="img.jpg" href="data:file/jpg;base64,{img_str}">Download result</a>'
	return href


st.write("""

# Images *booster*!

### Upload your photo and transform it with filters!


""")
with st.sidebar:
    choose = option_menu("Menu", ["About", "Filters", "Face detection", "Anime(BETA!)"],
                         icons=['info-lg', 'card-image', 'person-bounding-box', 'bandaid'],
                         menu_icon="app-indicator", default_index=0, 
                     #orientation = "Vertical"
    )



#Add file uploader to allow users to upload photos
uploaded_file = st.file_uploader("", type=['jpg','png','jpeg']) 

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

if choose == "About":
    st.write("""
    
    # Welcome!
    
    This is a program that let you experiment with your images and some features of the image processing.
    It was made to let you experiment all by yourself!
    
    The first tab is a recollection of some basic filters, they let you change the colors and saturation of your images.
    
    The second tab use a face-recognition system to modify deeper the layers of the images.
    
    And the last tab combine the machine-learning methods and offers you a way to visualize how you've be in an *anime* series. 
    (This part is in beta mode, so be patient, we know its limitations and will update it!)
    
    So, go ahead...
    
    # EXPERIMENT!
    
    
    
    """)

#Add 'before' and 'after' columns

if uploaded_file is not None:

    if choose == "Filters":
        image = load_img(uploaded_file)

        col1, col2 = st.columns( [0.5, 0.5])
        with col1:
            st.markdown('<p style="text-align: center;">Before</p>', unsafe_allow_html=True)
            st.image(image,width=300)  

        with col2:
            st.markdown('<p style="text-align: center;">After</p>',unsafe_allow_html=True) 
            filter = st.sidebar.radio('Filters:', ['Original','Gray Image', 'Black and White', 'Pencil Sketch', 'Blur Effect'])
            if filter == 'Gray Image':
                converted_img = np.array(image.convert('RGB'))
                gray_scale = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
                st.image(gray_scale, width=300)
            elif filter == 'Black and White':
                converted_img = np.array(image.convert('RGB'))
                gray_scale = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
                slider = st.sidebar.slider('Adjust the intensity', 1, 255, 127, step=1)
                (thresh, blackAndWhiteImage) = cv2.threshold(gray_scale, slider, 255, cv2.THRESH_BINARY)
                st.image(blackAndWhiteImage, width=300)
            elif filter == 'Pencil Sketch':
                converted_img = np.array(image.convert('RGB')) 
                gray_scale = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
                inv_gray = 255 - gray_scale
                slider = st.sidebar.slider('Adjust the intensity', 25, 255, 125, step=2)
                blur_image = cv2.GaussianBlur(inv_gray, (slider,slider), 0, 0)
                sketch = cv2.divide(gray_scale, 255 - blur_image, scale=256)
                st.image(sketch, width=300) 
            elif filter == 'Blur Effect':
                converted_img = np.array(image.convert('RGB'))
                converted_img = cv2.cvtColor(converted_img, cv2.COLOR_RGB2BGR)
                slider = st.sidebar.slider('Adjust the intensity', 5, 81, 33, step=2)
                blur_image = cv2.GaussianBlur(converted_img, (slider,slider), 0, 0)
                st.image(blur_image, channels='BGR', width=300)
            #elif filter == 'Face Detector':

            else: 
                st.image(image, width=300)         
    
    elif choose == "Face detection":
        image = load_img(uploaded_file)

        col1, col2 = st.columns( [0.5, 0.5])
        with col1:
            st.markdown('<p style="text-align: center;">Before</p>', unsafe_allow_html=True)
            st.image(image,width=300)
        with col2:
            st.markdown('<p style="text-align: center;">After</p>',unsafe_allow_html=True) 
            filter = st.sidebar.radio('Covert your photo to:', ['Original','Detection', 'Censor face', 'Censor no-face'
                                                               #, 'Little hat'
                                                               ])
            if filter == 'Detection':
                converted_img = np.array(image.convert('RGB'))
                converted_img = cv2.cvtColor(converted_img, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(converted_img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces)>0:
                    x, y, w, h = faces[0]
                    cv2.rectangle(converted_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = converted_img[y:y+h, x:x+w]
                    eyes = eye_cascade.detectMultiScale(roi_gray)
                    for (ex,ey,ew,eh) in eyes:
                        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                    converted_img[y:y+h, x:x+w] = roi_color
                    st.image(converted_img, channels='BGR', width=300)
                    converted_img = array_to_img(converted_img)
                    st.markdown(get_image_download_link(converted_img), unsafe_allow_html=True)
                elif len(faces)==0:
                    st.image(image, width=300)
                    st.write('No face detected')
            elif filter == 'Censor face':
                converted_img = np.array(image.convert('RGB'))
                gray = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces)>0:
                    x, y, w, h = faces[0]
                    p = int((0-x)*0.1 + (0-y)*0.10)
                    roi = converted_img[(y - p):(y + p)+h, (x - p):(x + p)+w]
                    slider = st.sidebar.slider('Adjust the blur', 3, 31, 15, step=2)
                    blur_roi = cv2.medianBlur(roi, slider)
                    converted_img[(y - p):(y + p)+h, (x - p):(x + p)+w] = blur_roi
                    st.image(converted_img, channels='RGB', width=300)
                    converted_img = array_to_img(converted_img)
                    st.markdown(get_image_download_link(converted_img), unsafe_allow_html=True)
                elif len(faces)==0:
                    st.image(image, width=300)
                    st.write('No face detected')
            elif filter == 'Censor no-face':
                converted_img = np.array(image.convert('RGB'))
                gray = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces)>0:
                    x, y, w, h = faces[0]
                    p = 0
                    roi = converted_img[(y - p):(y + p)+h, (x - p):(x + p)+w]
                    slider = st.sidebar.slider('Adjust the blur', 3, 31, 15, step=2)
                    converted_img = cv2.medianBlur(converted_img, slider)
                    converted_img[(y - p):(y + p)+h, (x - p):(x + p)+w] = roi
                    st.image(converted_img, channels='RGB', width=300)
                    converted_img = array_to_img(converted_img)
                    st.markdown(get_image_download_link(converted_img), unsafe_allow_html=True)
                elif len(faces)==0:
                    st.image(image, width=300)
                    st.write('No face detected')
#            elif filter == 'Little hat':
#                converted_img = np.array(image.convert('RGB'))
#                converted_img = cv2.cvtColor(converted_img, cv2.COLOR_RGB2BGR)
#                gray = cv2.cvtColor(converted_img, cv2.COLOR_BGR2GRAY)
#                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#                x, y, w, h = faces[0]
#                url_response = urlopen('https://www.kindpng.com/picc/m/31-315292_one-piece-straw-hat-png-transparent-png.png')
#                hat = cv2.imdecode(np.array(bytearray(url_response.read()), dtype=np.uint8), 1)
#                hat = cv2.resize(hat, (int(w*1.5), int(converted_img.shape[1]*0.50)))
#                img2gray2 = cv2.cvtColor(hat, cv2.COLOR_BGR2GRAY)
#                ret, mask = cv2.threshold(img2gray2, 200, 255, cv2.THRESH_BINARY_INV)
#                mask_inv = cv2.bitwise_not(mask)
#                roi = converted_img[(y+(int(0.45*hat.shape[0])))-hat.shape[0]:y+int(0.45*hat.shape[0]), x-int(x*0.98):x+2*(int(hat.shape[1]*0.50))]
#                img1_bg = cv2.bitwise_and(roi, roi, mask = mask_inv)
#                img2_fg = cv2.bitwise_and(hat, hat, mask = mask)
#                dst = cv2.add(img1_bg,img2_fg)
#                converted_img[(y+(int(0.45*hat.shape[0])))-hat.shape[0]:y+int(0.45*hat.shape[0]), x-int(x*0.98):x+2*(int(hat.shape[1]*0.50))] = dst
#                st.image(converted_img, channels='BGR', width=300)
#                converted_img = array_to_img(converted_img)
#                st.markdown(get_image_download_link(converted_img), unsafe_allow_html=True)
            else:
                st.image(image, width=300)
    
    elif choose == "Anime(BETA!)":
        image = load_img(uploaded_file)

        col1, col2 = st.columns( [0.5, 0.5])
        with col1:
            st.markdown('<p style="text-align: center;">Before</p>', unsafe_allow_html=True)
            st.image(image,width=300)
        with col2:
            st.markdown('<p style="text-align: center;">After</p>',unsafe_allow_html=True) 
            filter = st.sidebar.radio('Anime filters:', ['Original','Model 1', 'Model 2'
                                                        # , 'Model 3'
                                                        ])
            if filter == 'Model 1':
                converted_img = image.resize((256, 256))
                model_filename = 'anime_model.h5'
                cust = {'InstanceNormalization': InstanceNormalization}
                model1 = load_model(model_filename, cust, compile=False)
                imgx_array = np.array([img_to_array(converted_img)])
                converted_img = array_to_img(model1.predict(imgx_array)[0])
                st.image(converted_img, width=300)
                converted_img = array_to_img(converted_img)
                st.markdown(get_image_download_link(converted_img), unsafe_allow_html=True)
            #if filter == 'Model 2':
             #   converted_img = image.resize((256, 256))
             #   checkpoint = torch.load(cp_dir+'/'+ "face2anime_netG_95.pt")
            elif filter == 'Model 2':
                converted_img = image.resize((256, 256))
                model_filename = 'model_real2anime50i100sNorm.h5'
                model1 = load_model(model_filename, compile=False)
                imgx_array = np.array([img_to_array(converted_img)])
                converted_img = array_to_img(model1.predict(imgx_array)[0])
                st.image(converted_img, width=300)
                converted_img = array_to_img(converted_img)
                st.markdown(get_image_download_link(converted_img), unsafe_allow_html=True)
            else:
                st.image(image, width=300)

