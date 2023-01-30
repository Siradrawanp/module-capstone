import pandas as pd
import matplotlib.pyplot as plt
import itertools
import datetime

from time import time
from sklearn.model_selection import train_test_split

from embedding import embedding_w2v
from manhattan_dist import manhattan_distance
from keras.utils import pad_sequences
from keras.models import Model, Sequential
from keras.layers import Input, Embedding, LSTM, Lambda, Dense
from keras.optimizers import Adadelta, Adam
from keras.callbacks import EarlyStopping
import keras.backend as K

train_csv = './train_data/df-train-v1.csv'
test_csv = './train_data/df-test.csv'

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

max_seq_length = 15

train_df, test_df, embeddings = embedding_w2v(train_df, test_df, max_seq_length)

#max_seq_length = max(train_df.answer1.map(lambda x: len(x)).max(),
#                     train_df.answer2.map(lambda x: len(x)).max(),
#                     test_df.answer1.map(lambda x: len(x)).max(),
#                     test_df.answer2.map(lambda x: len(x)).max())

validation_size = int(len(train_df) * 0.33)
training_size = len(train_df) - validation_size

X = train_df[['answer1', 'answer2']]
Y = train_df['is_duplicate']

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

X_train = {'left': X_train.answer1, 'right': X_train.answer2}
X_validation = {'left': X_validation.answer1, 'right': X_validation.answer2}
X_test = {'left': test_df.answer1, 'right': test_df.answer2}

Y_train = Y_train.values
Y_validation = Y_validation.values

for dataset, side in itertools.product([X_train, X_validation], ['left', 'right']):
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)

#def manhattan_distance(left, right):
#    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

n_hidden = 50
gradient_clipping_norm = 1.25
batch_size = 1024 * 2
n_epoch = 200

left_input = Input(shape=(max_seq_length,), dtype='int32')
right_input = Input(shape=(max_seq_length,), dtype='int32')

embedding_layer = Embedding(len(embeddings), 400, weights=[embeddings], input_length=max_seq_length)

encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)

shared_lstm = LSTM(n_hidden, activation='tanh', recurrent_dropout=0.2)

left_output = shared_lstm(encoded_left)
right_output = shared_lstm(encoded_right)

malstm_distance = Lambda(function=lambda x: manhattan_distance(x[0], x[1]),output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

malstm = Model([left_input, right_input], [malstm_distance])

optimizer = Adam(clipnorm=gradient_clipping_norm, learning_rate=0.0001)

malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

training_start_time = time()

malstm_trained = malstm.fit([X_train['left'], X_train['right']], Y_train, batch_size=batch_size, epochs=n_epoch, shuffle=True, callbacks=[early_stopping], verbose=1,
                            validation_data=([X_validation['left'], X_validation['right']], Y_validation))

malstm.save('./model/SiameseLSTM_n1_10.h5')

print(str(malstm_trained.history['val_accuracy'][-1])[:6] +
      "(max: " + str(max(malstm_trained.history['val_accuracy']))[:6] + ")")
print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))


# Plot accuracy
plt.plot(malstm_trained.history['accuracy'])
plt.plot(malstm_trained.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot loss
plt.plot(malstm_trained.history['loss'])
plt.plot(malstm_trained.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()