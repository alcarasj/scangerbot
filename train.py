import nltk
import json
import pickle
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer


MODEL_DIR = './model'
IGNORE_WORDLIST = ['?', '!']
EPOCHS = 200
BATCH_SIZE =5


def create_training_data():
	lemmatizer = WordNetLemmatizer()
	data_file = open('intents.json').read()
	intents = json.loads(data_file)
	words = []
	documents = []
	classes = []
	for intent in intents['intents']:
		for pattern in intent['patterns']:
			w = nltk.word_tokenize(pattern)
			words.extend(w)
			documents.append((w, intent['tag']))
			if intent['tag'] not in classes:
				classes.append(intent['tag'])


	words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in IGNORE_WORDLIST]
	words = sorted(list(set(words)))
	classes = sorted(list(set(classes)))
	print(len(documents), "Documents")
	print(len(classes), "Classes", classes)
	print(len(words), "Unique lemmatized words", words)
	pickle.dump(words, open('%s/words.pkl' % MODEL_DIR,'wb'))
	pickle.dump(classes, open('%s/classes.pkl' % MODEL_DIR,'wb'))
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
	train_x = list(training[:, 0])
	train_y = list(training[:, 1])
	print("Training data created.")
	return (train_x, train_y)

def train(data):
	train_x = data[0]
	train_y = data[1]

	model = Sequential()
	model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(len(train_y[0]), activation='softmax'))
	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	hist = model.fit(np.array(train_x), np.array(train_y), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
	model.save('%s/model.h5' % MODEL_DIR, hist)

def main():
	data = create_training_data()
	train(data)

if __name__ == '__main__':
	main()