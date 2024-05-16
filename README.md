### While you can directly execute models using their framework APIs, Triton Inference Server offers significant advantages for streamlined, efficient, and scalable deployment:

- **Run multiple models concurrently** on GPUs for better throughput.
- **Dynamic batching** automatically optimizes inference requests for GPUs.
- **Upgrade models on the fly** without restarting Triton or client apps.
- **Dockerized deployment** simplifies deployment anywhere (on-prem/cloud).
- **Supports multiple frameworks** (TensorRT, TensorFlow, PyTorch, ONNX).
- **GPU & CPU acceleration** for flexibility based on your needs.

In this repository, I wanted to implement a simple web application using docker compose.

For the UI part, I used Streamlit:
![image](https://github.com/AntonioConsiglio/triton_server/assets/77753494/a133292f-8f33-4808-9241-3470799704df)

I have implemented 3 types of models, that is, the same model but using different backends:
  - Python and Pytorch
  - Onnxruntime
  - TensorRT

I have connected a PostgreSQL database to save the history of the current session (with a limit of up to 1 hour)
