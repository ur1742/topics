"""
OpenAlex Research Database Web Application
Функционал:
1. Поиск и скачивание записей из OpenAlex API
2. Анализ статистики публикаций
3. Классификация тематик с помощью нейросети
"""

from flask import Flask, render_template, request, jsonify, send_file
import requests
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import json
import os
from datetime import datetime
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max limit

# Глобальные переменные для хранения данных
works_data = []
classifier_model = None
vectorizer = None
label_encoder = None

# OpenAlex API базовый URL
OPENALEX_BASE_URL = "https://api.openalex.org"


def search_openalex(query, filter_type=None, per_page=20):
    """Поиск работ в базе OpenAlex"""
    try:
        endpoint = f"{OPENALEX_BASE_URL}/works"
        params = {
            'search': query,
            'per_page': min(per_page, 100),  # Максимум 100 записей за запрос
            'select': 'id,title,abstract_inverted_index,publication_year,cited_by_count,authorships,topics,open_access'
        }
        
        if filter_type:
            params['filter'] = filter_type
        
        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        results = data.get('results', [])
        
        # Обработка абстрактов (инвертированный индекс)
        for work in results:
            if 'abstract_inverted_index' in work and work['abstract_inverted_index']:
                work['abstract'] = reconstruct_abstract(work['abstract_inverted_index'])
            else:
                work['abstract'] = ''
        
        return results, len(results)
    
    except Exception as e:
        print(f"Error searching OpenAlex: {e}")
        return [], 0


def reconstruct_abstract(inverted_index):
    """Восстановление абстракта из инвертированного индекса"""
    if not inverted_index:
        return ""
    
    words_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            words_positions.append((pos, word))
    
    words_positions.sort(key=lambda x: x[0])
    return ' '.join([word for _, word in words_positions])


def analyze_statistics(works):
    """Анализ статистики по работам"""
    if not works:
        return {}
    
    stats = {
        'total_works': len(works),
        'years_distribution': {},
        'citations_distribution': {},
        'open_access_count': 0,
        'avg_authors': 0,
        'top_topics': {}
    }
    
    years = []
    citations = []
    authors_count = []
    topics_list = []
    
    for work in works:
        # Год публикации
        year = work.get('publication_year')
        if year:
            years.append(year)
            stats['years_distribution'][str(year)] = stats['years_distribution'].get(str(year), 0) + 1
        
        # Цитирования
        cites = work.get('cited_by_count', 0)
        citations.append(cites)
        
        # Открытый доступ
        if work.get('open_access', {}).get('is_oa', False):
            stats['open_access_count'] += 1
        
        # Авторы
        authorships = work.get('authorships', [])
        authors_count.append(len(authorships))
        
        # Темы
        for topic in work.get('topics', []):
            topic_name = topic.get('display_name', 'Unknown')
            topics_list.append(topic_name)
    
    # Статистика по цитированиям
    if citations:
        stats['citations_distribution'] = {
            'min': min(citations),
            'max': max(citations),
            'mean': round(np.mean(citations), 2),
            'median': round(np.median(citations), 2)
        }
    
    # Среднее количество авторов
    if authors_count:
        stats['avg_authors'] = round(np.mean(authors_count), 2)
    
    # Топ темы
    from collections import Counter
    if topics_list:
        topic_counts = Counter(topics_list)
        stats['top_topics'] = dict(topic_counts.most_common(10))
    
    return stats


def prepare_training_data(works):
    """Подготовка данных для обучения классификатора"""
    texts = []
    labels = []
    
    for work in works:
        title = work.get('title', '')
        abstract = work.get('abstract', '')
        text = f"{title} {abstract}".strip()
        
        if text:
            texts.append(text)
            
            # Используем темы как метки
            topics = work.get('topics', [])
            if topics:
                primary_topic = topics[0].get('display_name', 'General')
                labels.append(primary_topic)
            else:
                labels.append('General')
    
    return texts, labels


def train_classifier(texts, labels):
    """Обучение нейросетевого классификатора"""
    global classifier_model, vectorizer, label_encoder
    
    if len(texts) < 5:
        return False, "Недостаточно данных для обучения (минимум 5 записей)"
    
    try:
        # Кодирование меток
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        
        # Векторизация текста
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        X = vectorizer.fit_transform(texts)
        
        # Обучение нейросети
        classifier_model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=42
        )
        
        classifier_model.fit(X.toarray(), encoded_labels)
        
        return True, f"Модель обучена на {len(texts)} примерах, {len(label_encoder.classes_)} категорий"
    
    except Exception as e:
        return False, f"Ошибка обучения: {str(e)}"


def classify_text(text):
    """Классификация текста с помощью обученной модели"""
    global classifier_model, vectorizer, label_encoder
    
    if classifier_model is None or vectorizer is None or label_encoder is None:
        return None, "Модель не обучена"
    
    try:
        X = vectorizer.transform([text])
        prediction = classifier_model.predict(X)
        probabilities = classifier_model.predict_proba(X)[0]
        
        predicted_class = label_encoder.inverse_transform(prediction)[0]
        
        # Получаем топ-3 категории с вероятностями
        class_indices = np.argsort(probabilities)[::-1][:3]
        top_classes = [
            {
                'category': label_encoder.inverse_transform([idx])[0],
                'probability': round(float(probabilities[idx]) * 100, 2)
            }
            for idx in class_indices
        ]
        
        return top_classes, None
    
    except Exception as e:
        return None, f"Ошибка классификации: {str(e)}"


@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')


@app.route('/api/search', methods=['POST'])
def api_search():
    """API endpoint для поиска"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        per_page = int(data.get('per_page', 20))
        filter_type = data.get('filter', '')
        
        if not query:
            return jsonify({'error': 'Запрос не может быть пустым'}), 400
        
        results, count = search_openalex(query, filter_type, per_page)
        
        global works_data
        works_data = results
        
        return jsonify({
            'success': True,
            'count': count,
            'results': results[:50]  # Возвращаем максимум 50 для отображения
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/statistics', methods=['GET'])
def api_statistics():
    """API endpoint для получения статистики"""
    global works_data
    
    if not works_data:
        return jsonify({'error': 'Нет данных для анализа. Сначала выполните поиск.'}), 400
    
    stats = analyze_statistics(works_data)
    return jsonify(stats)


@app.route('/api/train', methods=['POST'])
def api_train():
    """API endpoint для обучения модели"""
    global works_data
    
    if not works_data:
        return jsonify({'error': 'Нет данных для обучения. Сначала выполните поиск.'}), 400
    
    texts, labels = prepare_training_data(works_data)
    
    if len(texts) == 0:
        return jsonify({'error': 'Не найдено работ с текстами для обучения'}), 400
    
    success, message = train_classifier(texts, labels)
    
    return jsonify({
        'success': success,
        'message': message,
        'samples_count': len(texts)
    })


@app.route('/api/classify', methods=['POST'])
def api_classify():
    """API endpoint для классификации текста"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Текст не может быть пустым'}), 400
        
        result, error = classify_text(text)
        
        if error:
            return jsonify({'error': error}), 400
        
        return jsonify({
            'success': True,
            'classifications': result
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download', methods=['GET'])
def api_download():
    """API endpoint для скачивания данных"""
    global works_data
    
    if not works_data:
        return jsonify({'error': 'Нет данных для скачивания'}), 400
    
    format_type = request.args.get('format', 'json')
    
    try:
        if format_type == 'csv':
            # Преобразование в DataFrame
            df_data = []
            for work in works_data:
                df_data.append({
                    'id': work.get('id', ''),
                    'title': work.get('title', ''),
                    'abstract': work.get('abstract', ''),
                    'publication_year': work.get('publication_year', ''),
                    'cited_by_count': work.get('cited_by_count', 0),
                    'open_access': work.get('open_access', {}).get('is_oa', False),
                    'authors_count': len(work.get('authorships', [])),
                    'topics': ', '.join([t.get('display_name', '') for t in work.get('topics', [])])
                })
            
            df = pd.DataFrame(df_data)
            output = io.BytesIO()
            df.to_csv(output, index=False, encoding='utf-8-sig')
            output.seek(0)
            
            return send_file(
                output,
                mimetype='text/csv',
                as_attachment=True,
                download_name=f'openalex_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            )
        
        else:  # JSON
            output = io.BytesIO()
            output.write(json.dumps(works_data, ensure_ascii=False, indent=2).encode('utf-8'))
            output.seek(0)
            
            return send_file(
                output,
                mimetype='application/json',
                as_attachment=True,
                download_name=f'openalex_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
