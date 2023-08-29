# MIT LICENSE
#
# Copyright 2023 Michael J. Reale
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

###############################################################################
# IMPORTS
###############################################################################

import sys
import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import cv2
import pandas
import sklearn
from huggingface_hub import model_info

###############################################################################
# MAIN
###############################################################################

def main():
    ###############################################################################
    # TENSORFLOW
    ###############################################################################

    a = tf.constant("Hello Tensorflow!")
    tf.print(a)
    print(tf.config.list_physical_devices('GPU'))           # Should list GPU devices
    print(tf.reduce_sum(tf.random.normal([1000, 1000])))    # Should print number tensor
    
    ###############################################################################
    # PYTORCH
    ###############################################################################
    
    b = torch.rand(5,3)
    print(b)
    print("Torch CUDA?:", torch.cuda.is_available())
    
    ###############################################################################
    # HUGGINGFACE_HUB
    ###############################################################################
    
    print(model_info('gpt2'))
    
    ###############################################################################
    # PRINT OUT VERSIONS
    ###############################################################################

    print("Tensorflow:", tf.__version__)    
    print("Torch:", torch.__version__)
    print("Numpy:", np.__version__)
    print("OpenCV:", cv2.__version__)
    print("Pandas:", pandas.__version__)
    print("Scikit-Learn:", sklearn.__version__)
        
    ###############################################################################
    # BASIC NEURAL NETWORK (TensorFlow)
    ###############################################################################
    
    # Load MNIST
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    def process_images(images):
        images = images.astype("float32")
        images /= 255.0
        images = (images - 0.5)*2.0 # Rescale from [0,1] to [-1,1]
        images = np.expand_dims(images, axis=-1) # Add channel dimension
        return images
        
    def process_labels(labels, num_classes):
        # Convert to one-hot vector
        one_hot_labels = np.eye(num_classes)[labels]
        return one_hot_labels
        
    x_train = process_images(x_train)
    x_test = process_images(x_test)
    
    y_train = process_labels(y_train, num_classes)
    y_test = process_labels(y_test, num_classes)
    
    print("x_train:", x_train.shape)
    print("x_test:", x_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)
    
    # Create quick TensorFlow NN model
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=x_train.shape[1:]))
    model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))	
        
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))	
        
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
        
    model.add(Dense(num_classes, activation='softmax'))
    
    model.summary()
    
    model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=10, batch_size=32)
    
    train_results = model.evaluate(x_train, y_train)
    test_results = model.evaluate(x_test, y_test)
    
    print("Training results:", train_results)
    print("Testing results:", test_results)    
    
if __name__ == "__main__": 
    main()