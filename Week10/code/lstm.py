from keras import layers, models
from keras.layers import Embedding
from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 10000
maxlen = 500
batch_size = 32
print('Loading data...')
(input_train0, y_train), (input_test0, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train0), 'train sequences')
print(len(input_test0), 'test sequences')
print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train0, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test0, maxlen=maxlen)

from keras.layers import LSTM

model = models.Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

review = [reverse_word_index.get(i-3, "?") for i in input_train0[0]]
print(input_train0[0])
print(review)

datas = ['I hate this movie','I very love this movie', 'if I can I will go to see ten times']
dataArr = [data.lower().split() for data in datas]
print(dataArr)
dataToInd = [[word_index.get(x)+3 for x in i] for i in dataArr]
print(dataToInd)
dataArr =  [[reverse_word_index.get(x-3, "?") for x in i]for i in dataToInd]
print(dataArr)
#model.predict_classes([indexData])