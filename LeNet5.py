from keras.layers import Dense, MaxPooling2D, Dropout, Flatten, Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
# from keras import optimizers


def train_cnn(X, y, DIM, index):

    model = Sequential()

    # Layer 1
    model.add(Convolution2D(12, (5, 5), strides=(1, 1), activation='relu', input_shape=(DIM, DIM, 1), kernel_initializer= 'he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(BatchNormalization())

    # Layer 2
    model.add(Convolution2D(25, (5, 5), activation='relu', kernel_initializer= 'he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(BatchNormalization())


    # Layer 3
    model.add(Flatten())
    model.add(Dense(180, activation='relu', kernel_initializer= 'he_normal'))
    model.add(Dropout(0.5))

    # Layer 4
    model.add(Dense(100, activation='relu', kernel_initializer= 'he_normal'))
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(10, activation='softmax', kernel_initializer= 'he_normal'))


    model.compile(loss='categorical_crossentropy', optimizer= 'adamax', metrics=['accuracy'])
    model.fit(X, y, batch_size=128, epochs=100)

    model.save('CNN_'+str(index)+'.h5')

    return model


# if __name__== "__main__":
#     main( )