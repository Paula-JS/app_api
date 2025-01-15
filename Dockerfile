FROM python:3.12-slim

RUN mkdir /templates

ADD /templates /templates
ADD app.py /
ADD requirements.txt /
ADD titanic_model.pkl /
ADD utils.py /

RUN pip install -r requirements.txt

CMD ["python", "app.py"]