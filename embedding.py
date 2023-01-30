import numpy as np

from gensim.models import KeyedVectors, Word2Vec
from nltk.corpus import stopwords

from extraction import keyword_extraction

def embedding_w2v(train_df, test_df, max_word):
    vocabulary = dict()
    inverse_vocabs = ['']
    stopword = set(stopwords.words('indonesian'))
    
    word2vec = KeyedVectors.load_word2vec_format("./model/wiki.id.case.vector")

    answer_column = ['answer1', 'answer2']

    for dataset in [train_df, test_df]:
        for index, row in dataset.iterrows():
            if  index != 0 and index % 200 == 0:
                print("{:,} sentences embedded.", format(index), flush=True)

            for answer in answer_column:
                q2n = []
                for word in keyword_extraction(row[answer], max_word):

                    if word in stopword and word not in word2vec:
                        continue

                    if word not in vocabulary:
                        vocabulary[word] = len(inverse_vocabs)
                        q2n.append(len(inverse_vocabs))
                        inverse_vocabs.append(word)
                    else:
                        q2n.append(vocabulary[word])
                
                dataset.at[index, answer] = q2n
    
    embedding_dim = 400
    embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)
    embeddings[0] = 0

    for word, index in vocabulary.items():
        if word in word2vec:
            embeddings[index] = word2vec.word_vec(word)
    del word2vec

    return train_df, test_df, embeddings