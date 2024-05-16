import requests
import json
from PIL import Image
import uuid

import streamlit as st
import numpy as np
from db import DBClass

from io import BytesIO

PALETTE = [0, 0, 0, 
            128, 0, 0, 
            0, 128, 0, 
            128, 128, 0,
            0, 0, 128,
            128, 0, 128,
            0, 128, 128, 
            128, 128, 128, 
            64, 0, 0,
            192, 0, 0,
            64, 128, 0, 
            192, 128, 0, 
            64, 0, 128,
            192, 0, 128,
            64, 128, 128, 
            192, 128, 128, 
            0, 64, 0, 
            128, 64, 0,
            0, 192, 0 , 
            128, 192, 0, 
            0, 64, 128,
            224,224,192] 
    # 0: (0, 0, 0),          # Background
    # 1: (128, 0, 0),        # Aeroplane
    # 2: (0, 128, 0),        # Bicycle
    # 3: (128, 128, 0),      # Bird
    # 4: (0, 0, 128),        # Boat
    # 5: (128, 0, 128),      # Bottle
    # 6: (0, 128, 128),      # Bus
    # 7: (128, 128, 128),    # Car
    # 8: (64, 0, 0),         # Cat
    # 9: (192, 0, 0),        # Chair
    # 10: (64, 128, 0),      # Cow
    # 11: (192, 128, 0),     # Dining Table
    # 12: (64, 0, 128),      # Dog
    # 13: (192, 0, 128),     # Horse
    # 14: (64, 128, 128),    # Motorbike
    # 15: (192, 128, 128),   # Person
    # 16: (0, 64, 0),        # Potted Plant
    # 17: (128, 64, 0),      # Sheep
    # 18: (0, 192, 0),       # Sofa
    # 19: (128, 192, 0),     # Train
    # 20: (0, 64, 128),      # TV/Monitor
    # 21: (224,224,192)      # Border

def _get_client_ip():
    from streamlit import runtime
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    session_id = get_script_run_ctx().session_id
    session_info = runtime.get_instance().get_client(session_id)
    return session_info.request.remote_ip
    
@st.cache_data(ttl=3600)
def _get_session(client_ip):
    return uuid.uuid4()

@st.cache_resource
def create_db(session_id):
    return DBClass(session_id)

#/--------- INITIALIZE SESSION AND DATABASE STORAGE BASED ON THE CLIENT IP ---------/
session_id = _get_session(_get_client_ip())

# Define the title
st.title("Triton Server based web application")
st.write(f"Session ID: {session_id}")
database = create_db(session_id)    

#/--------- SELECT THE MODEL TO USE ---------/
url = None
# Available models
options = ['BiSeNetV2', 'ONNX_BiSeNetV2', 'TRT_BiSeNetV2','RTFormer-B']

# Creare un menu a tendina
selection = st.selectbox('Select the model:', options)

# Utilizzare la scelta fatta dall'utente
if selection == 'BiSeNetV2':
    url = "http://triton_server:8000/v2/models/bisenetv2_service/infer"
elif selection == 'ONNX_BiSeNetV2':
    url = "http://triton_server:8000/v2/models/bisenetv2_onnx_service/infer"
elif selection == 'TRT_BiSeNetV2':
    url = "http://triton_server:8000/v2/models/bisenetv2_trt_service/infer"
elif selection == 'RTFormer-B':
    url = "http://triton_server:8000/v2/models/rtformer_service/infer"

#/--------- LOAD IMAGE TO PROCESS ---------/

st.write("Load the image to get your result! 😊 ")
image_loading = st.file_uploader("Upload Image",type=["jpg","jpeg","png"])

#/--------- SHOW IMAGE LOADED AND MASK RESULT ---------/
col1, col2 = st.columns(2)

if image_loading is not None:
    with col1:
        st.image(image_loading, caption='Uploaded Image.', use_column_width=True)

#/--------- SEND REQUEST TO TRITON---------/

def read_response_and_plot(response,origin_shape):
    output = response.get("outputs")[0]

    shape = output.get("shape")
    image = np.array(output.get("data")).reshape(shape).squeeze().astype(np.uint8)
    image = Image.fromarray(image).resize(origin_shape).convert("P")
    image.putpalette(PALETTE)
    return image

# When 'Submit' is selected
if st.button("Submit") and url is not None:

    image = Image.open(image_loading).convert("RGB")
    origin_shape = image.size
    image = image.resize((512,512))
    image = np.array(image).transpose(2,0,1)/255.0
    # Inputs to ML model
    inputs = {
    "inputs": [
        {
            "name": "image",
            "shape": [1, 3, 512, 512],
            "datatype": "FP32",
            "data": image.tolist()
        }
    ]
    }
    # Posting inputs to ML API
    response = requests.post(url,
                            headers={"Content-Type": "application/json"},
                            data=json.dumps(inputs))
    
    if response.status_code == 200:
        json_response = response.json()
        prediction = read_response_and_plot(json_response,origin_shape)
        with col2:
            st.image(prediction, caption='Prediction Mask', use_column_width=True)
        database.save_result(prediction)
    else:
        print("Error:", response.text)

#/--------- FETCH AND DISPLA THE LATEST 5 RESULTS---------/
st.subheader("Latest Results")
latest_results = database.fetch_latest_results()

if latest_results:
    cols = st.columns(5)
    for i, img_data in enumerate(latest_results):
        with cols[i]:
            img = Image.open(BytesIO(img_data))
            st.image(img, width=100)
            if st.button(f'Show Image {i + 1}', key=i):
                st.image(img, caption=f'Full Image {i + 1}',width=640)
