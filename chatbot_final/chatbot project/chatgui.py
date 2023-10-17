import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
import pickle
import numpy as np
import json
import random
from keras.models import load_model
from PIL import ImageTk
from tkinter import *

lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


def clean_up_sentence(sentence):
    # tokenizuj wzorzec - podziel słowa na tablicę
    sentence_words = nltk.word_tokenize(sentence)
    # przetwórz każde słowo - utwórz krótką formę słowa
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# zwróć tablicę bag of words: 0 lub 1 dla każdego słowa w worku, które występuje w zdaniu
def bow(sentence, words, show_details=True):
    # tokenizuj wzorzec
    sentence_words = clean_up_sentence(sentence)
    # Bag of Words (worek słów) - macierz N słów, macierz słownika
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # przypisz 1, jeśli obecne słowo znajduje się na danej pozycji w słowniku
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))


def predict_class(sentence, model):
    # odfiltruj predykcje poniżej określonego progu
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sortuj według siły prawdopodobieństwa
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    global result
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


def send(event=None):
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        res = chatbot_response(msg)
        ChatLog.image_create(END, image=bot_icon)
        ChatLog.insert(END, " " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

def clear_history():
    ChatLog.config(state=NORMAL)
    ChatLog.delete("1.0", END)
    ChatLog.config(state=DISABLED)

base = Tk()
base.title("Bank Assistant")
base.geometry("500x600")
base.resizable(width=FALSE, height=FALSE)
base.iconbitmap("bank-assistant.ico")

ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial")
ChatLog.config(state=DISABLED)

scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="10", height=5,
                    bd=0, bg="#808080", activebackground="#3c9d9b", fg='#ffffff',
                    command=send)

ClearButton = Button(base, font=("Verdana", 12, 'bold'), text="Clear", width="10", height=2,
                    bd=0, bg="#c0392b", activebackground="#a93226", fg='#ffffff',
                    command=clear_history)

EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")
EntryBox.bind("<Return>", send)

scrollbar.place(x=476, y=6, height=486)
ChatLog.place(x=6, y=6, height=486, width=470)
EntryBox.place(x=6, y=501, height=90, width=365)
SendButton.place(x=371, y=501, height=45)
ClearButton.place(x=371, y=546, height=45)


# Загрузка иконки из файла
bot_icon = ImageTk.PhotoImage(file="small-bank-assistant.jpg")

# Вывод приветствия и списка категорий первым сообщением
ChatLog.config(state=NORMAL)
ChatLog.image_create(END, image=bot_icon)
res = " Hello! 👋 You can ask me questions related to the following categories:\n\n"
categories = [
    "🔹 Login",
    "🔹 Registration",
    "🔹 Bank Search",
    "🔹 Bank Contact Information",
    "🔹 Account Balance",
    "🔹 Account Statement",
    "🔹 Transfer Funds",
    "🔹 Password Reset",
    "🔹 Card Services",
    "🔹 Customer Support"
]
categories_text = '\n'.join(categories)
res += categories_text
ChatLog.insert(END, res + '\n\n')
ChatLog.config(state=DISABLED)
ChatLog.yview(END)

ChatLog.image = bot_icon

base.mainloop()