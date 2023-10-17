import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam
import random
import time
import matplotlib.pyplot as plt

lemmatizer = WordNetLemmatizer()

words=[]
classes = []
documents = []
ignore_words = ['?', '!']

data_file = open('intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])

# Model z optymalizatorem SGD
model_sgd = Sequential()
model_sgd.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model_sgd.add(Dropout(0.5))
model_sgd.add(Dense(64, activation='relu'))
model_sgd.add(Dropout(0.5))
model_sgd.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model_sgd.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

start_sgd = time.time()
hist_sgd = model_sgd.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
end_sgd = time.time()
model_sgd.save('chatbot_model_sgd.h5', hist_sgd)

# Model z optymalizatorem Adam
model_adam = Sequential()
model_adam.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model_adam.add(Dropout(0.5))
model_adam.add(Dense(64, activation='relu'))
model_adam.add(Dropout(0.5))
model_adam.add(Dense(len(train_y[0]), activation='softmax'))

adam = Adam(learning_rate=0.01)
model_adam.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

start_adam = time.time()
hist_adam = model_adam.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
end_adam = time.time()
model_adam.save('chatbot_model_adam.h5', hist_adam)

print("Czas treningu SGD: ", end_sgd - start_sgd)
print("Czas treningu ADAM: ", end_adam - start_adam)

# Wykres wartości funkcji straty i dokładności treningowej
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 200), hist_sgd.history["loss"], label="strata_trening_sgd")
plt.plot(np.arange(0, 200), hist_sgd.history["accuracy"], label="dokładność_trening_sgd")
plt.plot(np.arange(0, 200), hist_adam.history["loss"], label="strata_trening_adam")
plt.plot(np.arange(0, 200), hist_adam.history["accuracy"], label="dokładność_trening_adam")
plt.title("Wartość funkcji straty i dokładności treningowej")
plt.xlabel("Numer epoki")
plt.ylabel("Strata/Dokładność")
plt.legend()
plt.show()
