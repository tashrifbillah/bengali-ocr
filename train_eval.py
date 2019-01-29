from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix
import numpy as np

from CNN_1 import train_cnn

def model_training(X_train, y_train, X_val, y_val, DIM, index):


    y_train_matrix = to_categorical(y_train, 10)
    model = train_cnn(X_train, y_train_matrix, DIM, index)

    # model evaluation
    prob = model.predict(X_val)
    temp = np.argmax(prob, axis=1).reshape(len(prob), 1)

    y_val = np.reshape(y_val, (len(y_val),))
    temp = np.reshape(temp, (len(temp),))

    # error calculation
    acc = np.sum(temp == y_val) / len(y_val)
    print("Average accuracy over all classes: %f" % acc)

    print("Confusion matrix")
    print(confusion_matrix(temp, y_val))

    return model

    # for i in range(10):
    #     ind1 = np.where(y_val == i)[0]
    #     acc = len(np.where(temp[ind1] == i)[0]) / len(ind1)
    #     print("Accuracy of class %d: %f" % (i, acc))


# if __name__== "__main__":
#     main( )