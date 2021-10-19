import json
import numpy as np
from pymorphy2 import MorphAnalyzer
from nltk.tokenize import WordPunctTokenizer
from string import digits, ascii_lowercase, punctuation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModel
import torch
import time
from gensim.models import KeyedVectors
import streamlit as st
import pickle
from scipy import sparse
import nltk


def preprocessing(text, preprocess):
    """Убирает стоп-слова, цифры и слова на латинице,
    возвращает леммы"""
    morph, tokenizer, stop = preprocess
    t = tokenizer.tokenize(text.lower())
    lemmas = [morph.parse(word)[0].normal_form for word in t
              if word not in punctuation and word not in stop and not set(word).intersection(digits)
              and not set(word).intersection(ascii_lowercase)]
    return ' '.join(lemmas)


def build_corpus(path):
    """Собирает леммы из файлов с текстами"""
    with open(path, 'r', encoding='utf-8') as f:
        texts = f.read().split('\n')
    return texts


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def indexation():
    nltk.download('stopwords')
    stop = set(stopwords.words("russian"))
    morph = MorphAnalyzer()
    wpt_tokenizer = WordPunctTokenizer()
    with open('tfidf.pickle', 'rb') as fin:
        tfidf_vectorizer = pickle.load(fin)
    with open('count.pickle', 'rb') as fin:
        count_vectorizer = pickle.load(fin)
    fasttext = KeyedVectors.load('araneum_none_fasttextcbow_300_5_2018.model')
    model = AutoModel.from_pretrained('./rubert-tiny')
    tokenizer = AutoTokenizer.from_pretrained('./rubert-tiny')
    bow_corpus = sparse.load_npz('bow_corpus.npz')
    tfidf_corpus = sparse.load_npz('tfidf_corpus.npz')
    bm_corpus = sparse.load_npz('bm_corpus.npz')
    bert_corpus = np.load('bert.npy')
    ft_corpus = np.load('ft_corpus.npy')
    matrices = {'bow': bow_corpus, 'tfidf': tfidf_corpus, 'fasttext': ft_corpus,
                'bm25': bm_corpus, 'bert': bert_corpus}
    models = {'bow': count_vectorizer, 'tfidf': tfidf_vectorizer, 'fasttext': fasttext,
              'bert': [model, tokenizer]}
    prepross = [morph, wpt_tokenizer, stop]
    return models, matrices, prepross


def get_emb(sent, models):
    model, tokenizer = models
    with torch.no_grad():
      enc = tokenizer(sent, return_tensors='pt')
      output = model(**enc, return_dict=True)
    vector = output.last_hidden_state[:,0,:].numpy()
    return vector


def sentence_embedding(sentence, model):
    """
    Складывает вектора токенов строки sentence
    """
    words = sentence.split()
    vectors = np.zeros(300)
    for i in words:
        if i in model:
            vectors += model[i]
    return np.expand_dims(vectors/len(words), axis=0)


def query_indexation(query, model, models):
    """Преобразовывает запрос в вектор"""
    if model == 'bert':
        return get_emb(query, models[model])
    elif model == 'fasttext':
        return sentence_embedding(query, models[model])
    elif model == 'bm25':
        return models['bow'].transform([query])
    else:
        return models[model].transform([query])


def count_cos(query, corpus):
    """Считает косинусную близость"""
    return cosine_similarity(query, corpus)[0]


def find_docs(query, corpus, answers, model, models, preprocess):
    """Выполняет поиск"""
    lemmas = preprocessing(query, preprocess)
    if lemmas:
        query_index = query_indexation(lemmas, model, models)
        sim = count_cos(query_index, corpus[model])
        ind = np.argsort(sim, axis=0)
        return np.array(answers)[ind][::-1].squeeze()
    else:
        return ['В Вашем запросе только цифры, пунктуация или латиница. Попробуйте еще раз!']


def search(matrices, corpus, models, preprocess):
    st.header('Найдите ответ на 1000 вопросов о любви!')
    query = st.text_input('Введите свой запрос: ')
    algorithm = st.selectbox('Выберите модель: ', ['tfidf', 'bm25', 'bert', 'bow', 'fasttext'])
    button = st.button('Искать', key='1')
    t = time.process_time()
    if button == 1:
        docs = find_docs(query, matrices, corpus, algorithm, models, preprocess)
        st.subheader('Вот что нам удалось узнать:')
        st.text('\n\n'.join(docs[:5]))
        st.text(f'Время выполнения запроса: {time.process_time() - t}')


def main():
    corpus = build_corpus('questions.txt')
    models, matrices, preprocess = indexation()
    search(matrices, corpus, models, preprocess)


if __name__ == '__main__':
    main()





