import random
import json
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
import pyglet
import gtts
import os
import speech_recognition as sr

"""Импортируем все необходимые библиотеки: работа с базой интентов через json, распознавание естесственной речи, 
её обработка, ввод-вывод звука и распознавание голоса"""

with open('BOT_CONFIG.json', 'r', encoding="utf8") as f:  # Загрузка конфига бота из файла
    BOT_CONFIG = json.load(f)


def clean(user_text):
    """Функция очистки введённого текста от "мусора"""""
    user_text = user_text.lower()
    cleaned_text = ''
    for char in user_text:
        if char in 'абвгдеёжзийклмпорстуфхцчшщъыьэюя':
            cleaned_text += char
    return cleaned_text


def get_intent(user_text):
    """Распознавание интента через расстояние Левенштейна"""
    for intent in BOT_CONFIG['intents'].keys():
        for example in BOT_CONFIG['intents'][intent]['examples']:
            cleaned_example = clean(example)
            cleaned_text = clean(user_text)
            if nltk.edit_distance(cleaned_example, cleaned_text) /\
                    max(len(cleaned_example), len(cleaned_text)) * 100 < 40:
                return intent
    return 'unknown_intent'


def user_input():
    """Функция распознавания голоса"""
    voice_recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            voice_recognizer.adjust_for_ambient_noise(source, duration=5)
            audio = voice_recognizer.listen(source)
            user_text = voice_recognizer.recognize_google(audio, lang='ru')
            return user_text
    except sr.UnknownValueError:
        print('Ошибка распознавания')


def voice_reply(text):
    """Функция голосового ответа"""
    print('Ответ: ', text)
    voice = gtts.gTTS(text, lang='ru')
    audio_file = 'bot_responce.mp3'
    voice.save(audio_file)
    sound = pyglet.resource.media(audio_file)
    sound.play()
    os.remove(audio_file)

"""Блок обучения через sklearn"""
X = []
y = []
for intent in BOT_CONFIG['intents']:
    for example in BOT_CONFIG['intents'][intent]['examples']:
        X.append(example)
        y.append(intent)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
clf = RidgeClassifier()
clf.fit(X_train_vectorized, y_train)
clf.score(X_train_vectorized, y_train)
clf.score(X_test_vectorized, y_test)


def get_intent_by_model(user_text):
    """Обработка интента"""
    vectorized_text = vectorizer.transform([user_text])
    return clf.predict(vectorized_text)[0]

def bot(user_text):
    """Тело бота"""
    intent = get_intent_by_model(user_text)
    return random.choice(BOT_CONFIG['intents'][intent]['responses'])


print('Укажите режим работы:'
          '\n1) Голосовой ввод'
          '\n2) Текстовый ввод')
choice = input()
# FIXME: Доделать выход из бота при типе интента "прощание"

if choice == "1":
    voice_reply('Пока не работает!')
    exit()
    # FIXME: Доделать голосовой ввод
    #user_text = user_input()
    #responce = bot(user_text)
    #voice_reply(responce)

elif choice == "2":
    while True:
        print('Введите текст >>>')
        user_text = input()
        responce = bot(user_text)
        voice_reply(responce)

else:
    print('Ошибка')
    exit()
