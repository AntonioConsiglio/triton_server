FROM python:3.9.4

RUN cd && mkdir /frontend_streamlit && cd /frontend_streamlit

WORKDIR /frontend_streamlit

COPY frontend/requirements.txt /frontend_streamlit/
RUN pip install --upgrade pip

RUN pip install -r /frontend_streamlit/requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "main.py"]