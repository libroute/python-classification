FROM python:3.6

RUN useradd -m user

WORKDIR /home/user
USER user

RUN pip install scikit-neuralnetwork --user
RUN pip uninstall --yes Theano
RUN pip install Theano==0.7 --user
RUN pip install numpy --user

RUN mkdir /home/user/data
WORKDIR /home/user/data
COPY classify.py ./
CMD ["python","classify.py"]
