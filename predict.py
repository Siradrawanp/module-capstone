import os
from time import time
import datetime
import itertools
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import keras.backend as K

from keras import models
from keras.utils import pad_sequences
from embedding import embedding_w2v
from manhattan_dist import manhattan_distance
from sort_to_csv import sort_to_csv
from output_csv import output_csv

path_answer = os.getcwd()+'\\answer_uas_td\*.txt'
path_key = './key_answer/key_uas2_td.txt'

predict_file = sort_to_csv(path_answer,path_key)
predict_test = './train_data/df-test.csv'

predict_start_time = time()
predict_df = pd.read_csv(predict_file)
test_df = pd.read_csv(predict_test)
max_seq_length = 15
predict_df, test_df, embeddings = embedding_w2v(predict_df, test_df, max_seq_length)

#print (predict_df)

X_pred = {'left': predict_df.answer1, 'right': predict_df.answer2}
print (X_pred)

for dataset, side in itertools.product([X_pred], ['left', 'right']):
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

assert X_pred['left'].shape == X_pred['right'].shape



model = models.load_model('./model/SiameseLSTM_n1_10.h5', custom_objects={"manhattan_distance": manhattan_distance})
model.summary()

prediction = model.predict([X_pred['left'], X_pred['right']], verbose=1)

#print (prediction)
for n in prediction:
    print("{:.3f}".format(float(n)))


output_file = "result_uasTD2_fin"
output_csv(path_answer, prediction, output_file)
print ("finished in {}".format(datetime.timedelta(seconds=time()-predict_start_time)))