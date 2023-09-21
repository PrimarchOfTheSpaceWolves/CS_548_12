import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def main():
    print("HELLO")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
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
    
    
        

if __name__ == "__main__":
    main()
    