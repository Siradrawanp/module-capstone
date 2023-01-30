import keras.backend as K

def manhattan_distance(left ,right):
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))
