import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.datasets import mnist,cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (InputLayer,
                                     Dense,
                                     Flatten,
                                     Conv2D,
                                     MaxPooling2D,
                                     Add)
from tensorflow.keras.models import Model
from tensorflow.keras import Input
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def main():
    print("HELLO")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    print("x_train:", x_train.shape)
    print("x_test:", x_test.shape)
    print("Image type:", x_train.dtype)
    
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)
    
    def preprocess_images(x):
        x = x.astype("float32")
        x /= 255.0
        if len(x.shape) <= 3:
            x = np.expand_dims(x, axis=-1)
        return x
    
    x_train = preprocess_images(x_train)
    x_test = preprocess_images(x_test)
    
    print("x_train AFTER:", x_train.shape)
    print("x_test AFTER:", x_test.shape)
    print("Image type AFTER:", x_train.dtype)
    
    '''
    model = Sequential()
    model.add(InputLayer(input_shape=x_train.shape[1:]))
    model.add(Conv2D(32, kernel_size=3,
                    padding="same", 
                    activation="relu"))
    model.add(Conv2D(32, kernel_size=3, 
                     padding="same", 
                     activation="relu"))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(64, kernel_size=3, 
                     padding="same", 
                     activation="relu"))
    model.add(Conv2D(64, kernel_size=3, 
                     padding="same", 
                     activation="relu"))
    model.add(MaxPooling2D(2))          
          
    model.add(Flatten())
    model.add(Dense(32, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    '''
    
    my_input = Input(shape=x_train.shape[1:])
    x = Conv2D(32, kernel_size=3,
                    padding="same", 
                    activation="relu")(my_input)
    
    x = Conv2D(32, kernel_size=3, 
                     padding="same", 
                     activation="relu")(x)
    
    x = MaxPooling2D(2)(x)
    
    alt_x = Dense(64)(x)
    
    x = Conv2D(64, kernel_size=3, 
                     padding="same", 
                     activation="relu")(x)
    x = Conv2D(64, kernel_size=3, 
                     padding="same", 
                     activation="relu")(x)
    
    x = Add()([x,alt_x])
    
    x = MaxPooling2D(2)(x)        
          
    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    my_output = Dense(10, activation="softmax")(x)
    
    model = Model(inputs=my_input, outputs=my_output)
    
    model.summary()
    
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    model.fit(x_train, y_train, batch_size=32, epochs=5)
    
    train_scores = model.evaluate(x_train, y_train, 
                                  batch_size=128)
    test_scores = model.evaluate(x_test, y_test,
                                 batch_size=128)
    
    print("TRAIN:", train_scores)
    print("TEST:", test_scores)
    
    
    
    
    
        

if __name__ == "__main__":
    main()
    