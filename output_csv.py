import os
import glob
import csv
import pandas as pd

from keras import models
from extraction import keyword_extraction
from embedding import embedding_w2v

def output_csv(path_answer, prediction, output_file):
    path_predict_result = './output/' + output_file + '.csv'
    predict_file = open(path_predict_result, 'w', encoding='UTF-8', newline='')
    header = ['id', 'nama', 'score']
    writer = csv.DictWriter(predict_file,fieldnames=header)
    writer.writeheader()

    index = 0
    nilai = 0
    score = []

    for n in prediction:
        
        # pengelompokan nilai
        if n == 1:
            nilai = 100
        elif 1 > n >= 0.5:
            nilai = 90 + (float("{:.3f}".format(float(n)))*10)
        elif 0.5 > n >= 0.1:
            nilai = 80 + (float("{:.3f}".format(float(n)))*10)
        elif 0.1 > n >= 0.05:
            nilai = 70 + (float("{:.3f}".format(float(n)))*10)
        elif 0.05 > n >= 0.01:
            nilai = 60 + (float("{:.3f}".format(float(n)))*10)
        elif 0.01 > n >= 0.005:
            nilai = 50 + (float("{:.3f}".format(float(n)))*10)
        elif 0.005 > n >= 0.001:
            nilai = 40 + (float("{:.3f}".format(float(n)))*10)
        else:
            nilai = 30 + (float("{:.3f}".format(float(n)))*10)                                            

        score.append(nilai)


    for filepath in glob.glob(path_answer):
        
        stdnt_name = os.path.basename(filepath)
        stdnt_name = os.path.splitext(stdnt_name)[0]
        
        rows = [{'id': index, 'nama': stdnt_name, 'score': score[index]}]
        index += 1

        writer.writerows(rows)
    
    predict_file.close
