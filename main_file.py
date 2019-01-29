from data_loading import train_data
from train_eval import model_training
import time
from prediction import test_prediction
import sys
from keras.models import load_model
import result_alignment


DIM= 180
dir_name= r"/home/pnl/Downloads/numta"
index= 9

def main():

    start_time= time.time()

    # train_data(dir_name, percentage of total data, DIM, train ration)
    X_train, Y_train, X_val, Y_val = train_data(dir_name, 100, DIM, 0.8)

    CNN= model_training(X_train, Y_train, X_val, Y_val, DIM, index)


    # Save the CNN architecture, and validation results in a .txt file
    orig_stdout = sys.stdout
    f = open('output'+str(index)+'.txt', 'w')
    # sys.stdout = f

    CNN.summary()
    test_prediction(dir_name, DIM, CNN, index)

    print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    # sys.stdout = orig_stdout
    f.close()

    result_alignment

if __name__== "__main__":
    main( )
