FROM python:3.12

WORKDIR /app

COPY requirements.txt ./

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev

# Instalar dependências Python
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Instalar datasets do nltk, incluindo o rslp
RUN python -m nltk.downloader punkt stopwords rslp

COPY . .

ENV FLASK_APP=__FinalXgboostPyMuPDF.py
ENV FLASK_ENV=development

EXPOSE 5001

CMD [ "flask", "run", "--host=0.0.0.0", "--port=5001" ]
