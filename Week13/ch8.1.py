#download the dataset
import keras
import numpy as np

path = keras.utils.get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()

def log(stringA):
    fp = open('ch8.1.log','a')
    fp.write(stringA)
    fp.close()

log("Corpus length:"+str(len(text))+'\n')

#extract sequences of "maxlen" chars
maxlen = 60
#sample a new sequence every "step" chars
step = 3
#hold the sequences
sentences = []
#hold the targets(the following char of a sequence)
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i:i+maxlen])
    next_chars.append(text[i+maxlen])
log('Number of sequences:'+str(len(sentences))+'\n')
log('sample at [2]:"'+ sentences[2] + '"\n')
log('sample at [3]:"'+ sentences[3] + '"\n')
log('next_chars at [2]:"'+ next_chars[2]+'"\n')
log('next_chars at [3]:"'+ next_chars[3]+'"\n')

chars = sorted(list(set(text)))
log('unique chars:'+str(len(chars))+'\n')
#using a for to combine the index and char to build a dict
char_indices = dict((char,chars.index(char))for char in chars)

log('vectorization\n')
#x is an array [len of sentences][maxlen][len of chars], one hot encode
#it is for samples
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
#y is [len of sentences][len of chars]
#it is for next_chars
y = np.zeros((len(sentences),len(chars)),dtype=np.bool)

#one hot encode
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i,t,char_indices[char]] = 1
        y[i,char_indices[next_chars[i]]] = 1

from keras import layers

model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars),activation='softmax'))

optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy',optimizer=optimizer)

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1,preds,1)
    return np.argmax(probas)

import random
import sys

for epoch in range(1,60):
    log('\nepochs: '+str(epoch)+'\n')
    model.fit(x,y,batch_size=128,epochs=1)
    start_index = random.randint(0,len(text))
    generated_text = text[start_index: start_index+maxlen]
    log('--- Generating with seed: "'+generated_text+'"\n')

    for temperature in [0.2,0.5,1.0,1.2]:
        log('------ temperature:'+ str(temperature)+'\n')
        log(generated_text)
        for i in range(400):
            sampled = np.zeros((1,maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0,t,char_indices[char]] = 1.
            
            preds = model.predict(sampled,verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]

            generated_text += next_char
            generated_text = generated_text[1:]

            log(next_char)
        
    if(epoch % 10 == 9):
        model.save("textGenEpoch"+str(epoch+1)+".h5")