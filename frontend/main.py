import requests
import json
from PIL import Image

import streamlit as st
import numpy as np

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


def read_response_and_plot(response,origin_shape):
    output = response.get("outputs")[0]

    shape = output.get("shape")
    image = np.array(output.get("data")).reshape(shape).squeeze().astype(np.uint8)
    image = Image.fromarray(image).resize(origin_shape).convert("P")
    image.putpalette(PALETTE)

    return image
# Define the title
st.title("Triton Server based web application")

url = None
# Available models
opzioni = ['BiSeNetV2', 'ONNX_BiSeNetV2', 'TRT_BiSeNetV2','RTFormer-B']

# Creare un menu a tendina
scelta = st.selectbox('Select the model:', opzioni)

# Utilizzare la scelta fatta dall'utente
if scelta == 'BiSeNetV2':
    url = "http://triton_server:8000/v2/models/bisenetv2_service/infer"
elif scelta == 'ONNX_BiSeNetV2':
    url = "http://triton_server:8000/v2/models/bisenetv2_onnx_service/infer"
elif scelta == 'TRT_BiSeNetV2':
    url = "http://triton_server:8000/v2/models/bisenetv2_trt_service/infer"
elif scelta == 'RTFormer-B':
    url = "http://triton_server:8000/v2/models/rtformer_service/infer"

st.write("Load the image to get your resutl")
image_loading = st.file_uploader("Upload Image",type=["jpg","jpeg","png"])

#/--------- SHOW IMAGE LOADED AND MASK RESULT ---------/
col1, col2 = st.columns(2)

if image_loading is not None:
    with col1:
        st.image(image_loading, caption='Uploaded Image.', use_column_width=True)


#/--------- SEND REQUEST TO TRITON---------/
# When 'Submit' is selected
if st.button("Submit") and url is not None:

    image = Image.open(image_loading).convert("RGB")
    origin_shape = image.size
    image = image.resize((512,512))
    image.save("input.png")
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
    else:
        print("Error:", response.text)
    
