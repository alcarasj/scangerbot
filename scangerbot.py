from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import nltk
import pickle
import numpy as np
import json
import random


ERROR_THRESHOLD = 0.25
MODEL_DIR = './model'

model = load_model('%s/model.h5' % MODEL_DIR)
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('%s/words.pkl' % MODEL_DIR,'rb'))
classes = pickle.load(open('%s/classes.pkl' % MODEL_DIR,'rb'))


def clean_up_sentence(sentence):
	sentence_words = nltk.word_tokenize(sentence)
	sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
	return sentence_words

def bow(sentence, words, show_details=True):
	sentence_words = clean_up_sentence(sentence)
	bag = [0] * len(words)  
	for s in sentence_words:
		for i,w in enumerate(words):
			if w == s: 
				bag[i] = 1
				if show_details:
					print ("Found in bag: %s" % w)
	return(np.array(bag))

def predict_class(sentence, model):
	p = bow(sentence, words, show_details=False)
	res = model.predict(np.array([p]))[0]
	results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
	results.sort(key=lambda x: x[1], reverse=True)
	return_list = []
	for r in results:
		return_list.append({ "intent": classes[r[0]], "probability": str(r[1]) })
	return return_list

def get_response(ints, intents_json):
	tag = ints[0]['intent']
	list_of_intents = intents_json['intents']
	for intent in list_of_intents:
		if intent['tag'] == tag:
			result = random.choice(intent['responses'])
			break
	return result

def chatbot_response(msg):
	ints = predict_class(msg, model)
	res = get_response(ints, intents)
	return res

def main():
	finished = False
	print('\n\n-----[CHAT]-----')
	while not finished:
		msg = input('[YOU]: ')
		if msg == 'exit':
			print('Exiting...')
			finished = True
		else:
			res = chatbot_response(msg)
			print('[SCANGERBOT]: %s' % res)

if __name__ == '__main__':
	main()