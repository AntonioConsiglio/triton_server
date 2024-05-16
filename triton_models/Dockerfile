# Use NVIDIA Triton Inference Server as base image
# FROM nvcr.io/nvidia/tritonserver:21.11-py3
FROM nvcr.io/nvidia/tritonserver:24.03-py3

# Set working directory
WORKDIR /models

COPY triton_models/requirements.txt /models

RUN pip3 install -r requirements.txt
# Expose Triton gRPC and HTTP ports
EXPOSE 8000
EXPOSE 8001
EXPOSE 8002

# Start Triton Server
# CMD ["tritonserver", "--model-repository=/models"]