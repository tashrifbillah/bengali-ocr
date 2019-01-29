from keras.layers import Dense, MaxPooling2D, Dropout, Flatten, Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
# from keras import optimizers


def train_cnn(X, y, DIM, index):

    model = Sequential()

    # Layer 1
    model.add(Convolution2D(16, (5, 5), strides=(2, 2), activation='relu', input_shape=(DIM, DIM, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization())

    # Layer 2
    model.add(Convolution2D(32, (3, 3), strides=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization())

    # Layer 3
    model.add(Convolution2D(64, (3, 3), activation='relu'))

    # Layer 4
    model.add(Convolution2D(256, (3, 3), activation='relu'))

    # Layer 5
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    # Layer 6
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(10, activation='softmax'))


    # sgd = optimizers.SGD(lr=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer= 'sgd', metrics=['accuracy'])
    model.fit(X, y, batch_size=32, epochs=100)

    model.save('CNN_'+str(index)+'.h5')

    return model


# if __name__== "__main__":
#     main( )
