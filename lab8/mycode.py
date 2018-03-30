from keras.layers import Dense, Activation, Embedding, Flatten, Input, Dropout, Conv1D, GlobalMaxPooling1D, LSTM
from keras.datasets import imdb
from __future__ import print_function
import numpy as np
np.random.seed(1337) 

from keras.preprocessing import sequence
from keras.models import Model, Sequential


max_features = 20000
maxlen = 80  # cut texts after this number of words 
batch_size = 32


(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print (X_train[0])

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


inputs = Input(shape=(maxlen,))
x = inputs
y = Embedding(max_features, 128, dropout=0.2)(x)
z = LSTM(32)(y)
h = Dense(1)(z)
predictions = Activation("sigmoid")(h)

model = Model(input=inputs, output=predictions)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=15,
          validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('score:', score)
print('accuracy:', acc)