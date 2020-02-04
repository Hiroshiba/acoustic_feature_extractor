FROM hiroshiba/hiho-deep-docker-base:miniconda-python3.7.5

RUN apt-get update && apt-get install -y swig libsndfile1-dev libasound2-dev && apt-get clean
RUN conda install -y cython numpy numba

WORKDIR /app

# install requirements
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# add source cord
COPY analyzer /app/analyzer
COPY extractor /app/extractor
COPY acoustic_feature_extractor /app/acoustic_feature_extractor
COPY tests /app/tests
COPY setup.py /app/setup.py
