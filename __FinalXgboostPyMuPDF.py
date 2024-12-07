import os
import tempfile
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import pickle
import fitz  # PyMuPDF
from flask import Flask, jsonify, request
import logging
import xgboost as xgb

app = Flask(__name__)

logging.basicConfig(filename='training_errors.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Configure stopwords and stemmer
stopwords = stopwords.words('portuguese')
stemmer = RSLPStemmer()

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_document = fitz.open(file)
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                text += page.get_text()
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
    return text

def preprocess_text(text):
    words = word_tokenize(text, language='portuguese')
    words = [stemmer.stem(word) for word in words if word not in stopwords]
    return ' '.join(words)

def save_data(X_train, X_test, y_train, y_test, file_path):
    data = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
    print(f'Data saved to {file_path}')

def train_model():
    sector_mapping = {'PAS': 0, 'PDA': 1, 'PPE': 2, 'PSE': 3, 'PTR': 4, 'PUMA': 5, 'PTA': 6}
    data_dir = './DirTrein'
    pdf_files = os.listdir(data_dir)
    documents, labels = [], []

    for file in tqdm(pdf_files, desc='Processing PDFs'):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(data_dir, file)
            text = extract_text_from_pdf(pdf_path)
            documents.append(text)
            sector_code = file.split('_')[0]
            sector_label = sector_mapping.get(sector_code)
            if sector_label is not None:
                labels.append(sector_label)
            else:
                print(f'Warning: Invalid sector code found in file {file}')

    df = pd.DataFrame({'documents': documents, 'labels': labels})
    X = df['documents']
    y = df['labels']

    pipeline = Pipeline([
        ('vect', CountVectorizer(max_features=10000)),
        ('clf', xgb.XGBClassifier(eval_metric='mlogloss'))
    ])

    print('Preprocessing training data...')
    X_prep = X.apply(preprocess_text)
    print('Preprocessing completed!')

    pipeline.fit(X_prep, y)

    X_train, X_test, y_train, y_test = train_test_split(X_prep, labels, test_size=0.2, random_state=42)
    predictions = pipeline.predict(X_test)
    report = classification_report(y_test, predictions)
    file_path = 'trainingXgboost.pkl'
    save_data(X_train, X_test, y_train, y_test, file_path)

    print('\nModel training data saved')
    print(report)
    
    return pipeline

def load_data_classificacao(file_path_class):
    if os.path.exists(file_path_class) and os.path.getsize(file_path_class) > 0:
        try:
            with open(file_path_class, 'rb') as file:
                data = pickle.load(file)
            print("Data loaded successfully.")
            return data
        except (EOFError, pickle.UnpicklingError):
            print("File is empty or corrupted. Retraining the model...")
            return None
    else:
        print("File does not exist. Retraining the model...")
        return None

def predict_classificacao(file_path_class, switch_case_class):
    initialPetition = extract_text_from_pdf(file_path_class)
    if initialPetition:
        X_prediction = [initialPetition]
        prediction_preprocessing = preprocess_text(X_prediction[0])
        prediction = pipeline_class.predict([prediction_preprocessing])
        specialized = switch_case_class.get(prediction[0], 'NOT FOUND')
        return specialized
    else:
        return "Failed to convert PDF to text."

# Load data and fit model; retrain if loading fails
loaded_data = load_data_classificacao('trainingXgboost.pkl')
pipeline_class = Pipeline([
    ('vect', CountVectorizer(max_features=10000)),
    ('clf', xgb.XGBClassifier(eval_metric='mlogloss'))
])

if loaded_data:
    X_train, X_test, y_train, y_test = loaded_data['X_train'], loaded_data['X_test'], loaded_data['y_train'], loaded_data['y_test']
    pipeline_class.fit(X_train, y_train)
else:
    pipeline_class = train_model()

@app.route('/treino', methods=['POST'])
def resposta():
    try:
        pipeline = train_model()
        global pipeline_class
        pipeline_class = pipeline
        return jsonify({'message': 'Model trained successfully!'})
    except Exception as e:
        logging.error("Falha ao treinar o modelo: %s", str(e))
        return jsonify({'error': 'Falha ao treinar IA'}), 500

@app.route('/classificar', methods=['POST'])
def resposta2():
    if 'uploaded_file' not in request.files:
        return jsonify({'error': 'Selecione um arquivo PDF.'}), 400
    
    file = request.files['uploaded_file']
    if file.filename == '':
        return jsonify({'error': 'Selecione um arquivo PDF.'}), 400

    if file and file.filename.endswith('.pdf'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            file_path = temp_file.name
            file.save(file_path)
        
        try:
            result = predict_classificacao(file_path, switch_case)
        finally:
            os.remove(file_path)
        
        return jsonify({'message': 'Classification successful!', 'classification_report': result})
    else:
        return jsonify({'error': 'O arquivo não é um PDF.'}), 400

def copiar_pdf_para_diretorio(arquivo_pdf, destino_diretorio, especializada):
    if not os.path.isdir(destino_diretorio):
        os.makedirs(destino_diretorio)
        print(f"O diretório {destino_diretorio} foi criado.")
    
    try:
        nome_arquivo, extensao = os.path.splitext(arquivo_pdf.filename)
        novo_nome_arquivo = f"{especializada}_{nome_arquivo}{extensao}"
        caminho_completo = os.path.join(destino_diretorio, novo_nome_arquivo)
        arquivo_pdf.save(caminho_completo)
        return "Arquivo enviado com sucesso"
    except Exception as e:
        return f"Ocorreu um erro ao salvar o arquivo PDF: {str(e)}"

@app.route('/ajustar', methods=['POST'])
def resposta3():
    try:
        if 'correcao_file' not in request.files:
            return jsonify({'message': 'É NECESSÁRIO ENVIAR UM ARQUIVO'}), 400
        
        arquivo_pdf = request.files['correcao_file']
        
        if not arquivo_pdf.filename.endswith('.pdf'):
            return jsonify({'message': 'O ARQUIVO NÃO É UM PDF.'}), 400

        input_especializada = request.form['especializada'].upper()

        if input_especializada == '':
            return jsonify({'message': 'DIGITE UMA ESPECIALIZADA.'}), 400 
            
        elif input_especializada not in ['PAS', 'PDA', 'PPE', 'PSE', 'PTR', 'PUMA', 'PTA']:
            return jsonify({'message': 'SIGLA INVÁLIDA'}), 400
        
        caminho_dir = './DirTrein'
        result = copiar_pdf_para_diretorio(arquivo_pdf, caminho_dir, input_especializada)
        
        return jsonify({'message': 'Document adjusted successfully!', 'classification_report': result})
    except Exception:
        return jsonify({'message': 'ERRO'}), 400

switch_case = {0: 'PAS', 1: 'PDA', 2: 'PPE', 3: 'PSE', 4: 'PTR', 5: 'PUMA', 6: 'PTA'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True, extra_files=['trainingXgboost.pkl'])
