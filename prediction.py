import pandas as pd
from PIL import Image
import os
import numpy as np
# from keras.models import load_model


def load_test_data(dir_name, DIM):

    datasets = ['a', 'b', 'c', 'd', 'e', 'f', 'auga', 'augc']

    temp = np.ndarray(shape=(0, DIM, DIM, 1))
    names= [ ]

    for data in datasets:

        cwd = os.getcwd()
        os.chdir(dir_name + "/testing-{0}/".format(data))
        nwd = os.getcwd( )

        file_names= os.listdir(nwd)
        names.append(file_names)


        for i in file_names:
            I = Image.open(i)
            C = I.resize((DIM, DIM))
            C = np.array(C.convert('L'), dtype=np.uint8).reshape((1, DIM, DIM, 1))

            temp = np.concatenate((temp, C), axis=0)

        os.chdir(cwd)

        print(data)

    flat_names = [item for sublist in names for item in sublist]

    return (temp, flat_names)




def test_prediction(dir_name, DIM, model, index):

    X_test, file_names= load_test_data(dir_name, DIM)

    # model = load_model('CNN_6.h5')

    prob = model.predict(X_test)
    y_test = np.argmax(prob, axis=1).flatten()


    d = {'key': file_names, 'label': y_test}
    df = pd.DataFrame(data=d)
    df.to_csv('prediction_'+str(index)+'.csv', index=False)


def main():

    test_prediction(r"/home/pnl/Downloads/numta", DIM= 180)


if __name__== "__main__":
    main( )