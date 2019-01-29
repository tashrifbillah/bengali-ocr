import pandas as pd
import numpy as np
import sys
from PIL import Image
from sklearn.model_selection import train_test_split



def set_formation(dir_name, file_names, DIM, data):


    X = np.zeros((len(file_names), DIM, DIM, 1))

    num = 0

    for i in file_names:

        I = Image.open(dir_name + "/training-{0}/".format(data) + i)
        C = I.resize((DIM, DIM))
        C = np.array(C.convert('L'), dtype=np.uint8).reshape((DIM, DIM, 1))
        X[num, :, :, :] = C

        num = num + 1

    return X


def train_data(dir_name, percent, DIM, train_ratio):

    datasets = ['a', 'b', 'c', 'd', 'e']

    X_train = np.ndarray(shape=(0, DIM, DIM, 1))
    Y_train = np.ndarray(shape=(0, 1))

    X_val = np.ndarray(shape=(0, DIM, DIM, 1))
    Y_val = np.ndarray(shape=(0, 1))


    for data in datasets:

        temp = pd.read_csv(dir_name + "/training-{0}.csv".format(data))

        ind= np.random.randint(0, temp.shape[0], round(temp.shape[0]*percent/100))
        ind_train, ind_val = train_test_split(ind, test_size= train_ratio)


        # Train set
        file_names = temp['filename'].values[ind_train]
        labels= temp['digit'].values[ind_train]

        X= set_formation(dir_name, file_names, DIM, data)
        Y = np.array(labels, dtype=np.uint8).reshape(len(labels),1)
        X_train = np.concatenate((X_train, X), axis=0)
        Y_train = np.concatenate((Y_train, Y), axis=0)


        # Validation set
        file_names = temp['filename'].values[ind_val]
        labels= temp['digit'].values[ind_val]

        X= set_formation(dir_name, file_names, DIM, data)
        Y = np.array(labels, dtype=np.uint8).reshape(len(labels),1)
        X_val = np.concatenate((X_val, X), axis=0)
        Y_val = np.concatenate((Y_val, Y), axis=0)



    return (X_train, Y_train, X_val, Y_val)


# if __name__== "__main__":
#     main( )

