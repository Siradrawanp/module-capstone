import os
import glob
import csv
import pandas as pd

from keras import models
from extraction import keyword_extraction
from embedding import embedding_w2v

def sort_to_csv(path_answer, path_key):
    path_predict_process = './output/temp_predict.csv'
    predict_file = open(path_predict_process, 'w', encoding='UTF-8', newline='')
    header = ['test_id', 'answer1', 'answer2']
    writer = csv.DictWriter(predict_file,fieldnames=header)
    writer.writeheader()

    index = 0

    for filepath in glob.glob(path_answer):
        
        stdnt_name = os.path.basename(filepath)
        stdnt_name = os.path.splitext(stdnt_name)[0]

        #print(stdnt_name)
        with open(filepath) as f:
            contents = f.read()
            #print(contents)
        
        with open(path_key) as key:
            f_key = key.read()
            #print(f_key)
        
        rows = [{'test_id': index, 'answer1': contents, 'answer2': f_key}]
        index += 1

        writer.writerows(rows)

    predict_file.close
    return path_predict_process