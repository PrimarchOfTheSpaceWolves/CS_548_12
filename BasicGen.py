import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.datasets import mnist,cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (InputLayer,
                                     Dense,
                                     Flatten,
                                     Conv2D,
                                     Dropout,
                                     MaxPooling2D,
                                     Concatenate,
                                     Conv2DTranspose,
                                     BatchNormalization,
                                     ReLU,
                                     LeakyReLU,
                                     Reshape,
                                     Lambda,
                                     Add)
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.applications.vgg19 import VGG19
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
from sklearn.model_selection import train_test_split

def make_simple_generator(noise_dim):
    init_input = Input(shape=(noise_dim,))
    x = init_input
    
    x = Dense(8*8*512, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Reshape((8,8,512))(x)
    
    x = Conv2DTranspose(256, 4, strides=2, 
                        padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2DTranspose(128, 4, strides=2,
                        padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2DTranspose(1, 4, padding="same",
                        activation="tanh")(x)
    
    
    model = Model(inputs=init_input, outputs=x)
    return model
    
def make_simple_discriminator():
    init_input = Input(shape=(32,32,1))
    x = init_input
    
    x = Conv2D(128, 4, padding="same", strides=2)(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(256, 4, padding="same", strides=2)(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(512, 4, padding="same", strides=2)(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Flatten()(x)
    x = Dense(1)(x)
    
    model = Model(inputs=init_input, outputs=x)
    return model
    

def main():
    noise_dim = 100
    gen = make_simple_generator(noise_dim)
    gen.summary()
    
    dis = make_simple_discriminator()
    dis.summary()
    
    # Load MNIST
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    def process_images(images):
        images = images.astype("float32")        
        images /= 255.0
        images = (images - 0.5)*2.0 # Rescale from [0,1] to [-1,1]
        images = np.expand_dims(images, axis=-1) # Add channel dimension
        images = tf.image.resize(images, (32,32)).numpy()
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
    
    loss_fn = tf.keras.losses.BinaryCrossentropy(
                    from_logits=True)
    
    def calc_gen_loss(fake_output):
        loss = loss_fn(tf.ones_like(fake_output), fake_output)
        return loss
    
    def calc_dis_loss(fake_output, real_output):
        fake_loss = loss_fn(tf.zeros_like(fake_output),
                            fake_output)
        real_loss = loss_fn(tf.ones_like(real_output), 
                            real_output)
        return fake_loss + real_loss
    
    gen_opt = tf.keras.optimizers.Adam(1e-4)
    dis_opt = tf.keras.optimizers.Adam(1e-4)
    
    @tf.function
    def train_step(images):
        batch_cnt = images.shape[0]
        noise = tf.random.normal((batch_cnt, noise_dim))
        
        with (tf.GradientTape() as gen_tape, 
              tf.GradientTape() as dis_tape):
            fakes = gen(noise, training=True)
            
            real_output = dis(images, training=True)
            fake_output = dis(fakes, training=True)
            
            gen_loss = calc_gen_loss(fake_output)
            dis_loss = calc_dis_loss(fake_output,
                                     real_output)
            
        gen_grads = gen_tape.gradient(gen_loss,
                                       gen.trainable_weights)
        dis_grads = dis_tape.gradient(dis_loss,
                                       dis.trainable_weights)
        
        gen_opt.apply_gradients(zip(gen_grads,
                                    gen.trainable_weights))
        dis_opt.apply_gradients(zip(dis_grads,
                                    dis.trainable_weights))
        
        return gen_loss, dis_loss
        
    train_ds = tf.data.Dataset.from_tensor_slices(x_train)
    epoch_cnt = 50
    batch_size = 32
    
    train_cnt = train_ds.cardinality()
    train_ds = train_ds.shuffle(train_cnt)
    train_ds = train_ds.batch(batch_size)
    
    rand_aug = Sequential([
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomTranslation(0.1, 0.1)
    ])
    
    def do_aug(x):
        x = rand_aug(x, training=True)
        return x
    
    base_train_ds = train_ds
    train_ds = train_ds.map(do_aug, 
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
          
    
    sample_noise = tf.random.normal([1, noise_dim])
    
    for epoch in range(epoch_cnt):
        print("Epoch", epoch)
        
        batch_index = 0
        ave_gen_loss = 0
        ave_dis_loss = 0
        
        for batch in train_ds:
            gen_loss, dis_loss = train_step(batch)
            
            gen_loss = gen_loss.numpy()
            dis_loss = dis_loss.numpy()
            
            if batch_index % 20 == 0:
                print("\t", "Batch", batch_index,
                      ":", gen_loss, ",", dis_loss)
                sample = gen(sample_noise, training=False)
                sample = sample[0].numpy()
                sample = cv2.resize(sample, (256,256))
                sample += 1.0
                sample /= 2.0
                cv2.imshow("GEN", sample)
                cv2.waitKey(30)
                            
            batch_index += 1
            ave_gen_loss += gen_loss
            ave_dis_loss += dis_loss
            
        ave_gen_loss /= batch_index
        ave_dis_loss /= batch_index
        
        print("AVERAGE:", ave_gen_loss, ave_dis_loss)
    cv2.destroyAllWindows()
            
                    
            
            
            
        

if __name__ == "__main__":
    main()
    
