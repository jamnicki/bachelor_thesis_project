FROM python:3.8

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY spacy-ner-al.py /app/spacy-ner-al.py

RUN pip install -r /tmp/requirements.txt

CMD ["python", "/src/spacy-ner-al.py"]
