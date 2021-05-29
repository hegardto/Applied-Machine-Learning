# -*- coding: utf-8 -*-

# -- Sheet --

# ## Introduction: Loading images from a directory


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

plt.style.use('seaborn')
%config InlineBackend.figure_format = 'retina'

data_gen = ImageDataGenerator(rescale= 1.0/255)

imgdir = 'a5_images'
img_size = 64
batch_size = 32

generator_train = data_gen.flow_from_directory(
        imgdir + '/train',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        classes=['other', 'car'],
        seed=12345,
        shuffle=True)

Xtrain, Ytrain = generator_train.next()

Xtrain.shape

Ytrain.shape

generator_validation = data_gen.flow_from_directory(
        imgdir + '/validation',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        classes=['other', 'car'],
        seed=12345,
        shuffle=True)

Xvalidate, Yvalidate = generator_validation.next()

plt.imshow(Xtrain[4]);

# ## Part 1: Training a convolutional neural network


def make_convnet(img_size_vertical, img_size_horizontal):
    
    model = keras.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(5,5), strides=(1,1), activation = 'relu', input_shape = (img_size_vertical, img_size_horizontal, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2,2), strides =(2,2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))
  	
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model

model = make_convnet(64, 64)

model_fit = model.fit_generator(generator_train, 
                  validation_data=generator_validation, 
                  steps_per_epoch=50,
                  validation_steps=18,
                  epochs=10)

plt.plot(model_fit.history['val_accuracy'])
plt.plot(model_fit.history['accuracy'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Validation accuracy', 'Train accuracy'])
plt.show()

plt.plot(model_fit.history['val_loss'])
plt.plot(model_fit.history['loss'])
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Validation loss', 'Train loss'])
plt.show()

# ## Part 2: Data augmentation


data_gen1 = ImageDataGenerator(rescale= 1.0/255, horizontal_flip=True, rotation_range=60)

imgdir = 'a5_images'
img_size = 64
batch_size = 32

generator_train = data_gen1.flow_from_directory(
        imgdir + '/train',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        classes=['other', 'car'],
        seed=12345,
        shuffle=True)

Xtrain, Ytrain = generator_train.next()

plt.imshow(Xtrain[4]);

Xtrain = Xtrain.reshape(batch_size, img_size, img_size, 3)

generator_validation = data_gen1.flow_from_directory(
        imgdir + '/validation',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        classes=['other', 'car'],
        seed=12345,
        shuffle=True)

model = make_convnet(64, 64)

generator_fit = model.fit_generator(generator_train, 
                  validation_data=generator_validation, 
                  steps_per_epoch=50,
                  validation_steps=18,
                  epochs=10)

# ## Interlude: Applying a pre-trained convolutional neural network


from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import decode_predictions, preprocess_input

vggmodel = applications.VGG16(weights='imagenet', include_top=True)

image = load_img(imgdir + '/train/car/0010.jpg', target_size=(224,224))
plt.imshow(image);
image_array = img_to_array(image)
image_array = preprocess_input(image_array)
image_array = image_array.reshape(1, 224, 224, 3)

pred = vggmodel.predict(image_array)

print(decode_predictions(pred))

# ## Part 3: Using VGG-16 as a feature extractor


def create_vgg16_features(filedir, img_size_vertical, img_size_horizontal, batch_size, train_val):
    
    feature_extractor = applications.VGG16(include_top=False, weights='imagenet',
                                        input_shape=(img_size_vertical, img_size_horizontal, 3))

    vgg_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    generator = vgg_data_gen.flow_from_directory(
            filedir + train_val,
            target_size=(img_size_vertical, img_size_horizontal),
            batch_size=batch_size,
            class_mode='binary',
            classes=['other', 'car'],
            seed=12345,
            shuffle=False)

    cnn_features = feature_extractor.predict(generator)

    with open(train_val+'.txt', 'wb') as f:
        np.save(f, cnn_features)

create_vgg16_features('a5_images/',64,64,32,'train')

create_vgg16_features('a5_images/',64,64,32,'validation')

def get_labels(n):
    return np.array([0]*(n//2) + [1]*(n//2))

def train_on_cnnfeatures(train_file, validation_file):
    
    with open(train_file,'rb') as f:
        X_train = np.load(f)

    print(X_train.shape)

    with open(validation_file,'rb') as f:
        X_validation = np.load(f)

    print(X_validation.shape)

    Y_train = get_labels(len(X_train))
    print(Y_train.shape)
    Y_validation = get_labels(len(X_validation))
    print(Y_validation.shape)
   
    model = keras.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model_fit = model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation),
                  steps_per_epoch=50,
                  epochs=20)
    

train_on_cnnfeatures('train.txt', 'validation.txt')

# The validation accuracy for 20 epochs and 50 steps per epoch was 0.9045 which is signifantly higher than for the CNN in part 2. We believe that the reason for this diffence is that the model is now analyzing picture features (metadata of the picture generated by the VGG model) instead of pixels. There is probably a lot more useful information to utilize for the neural network from data stating that the picture probably contains an ambulance or a fire truck than data from just a pixel. The neural network learns that some picture features, which are describing words generated by the VGG model, are more likely to be connected to one of the labels (car or other) which helps it to analyze the picture. The difference between computing on describing words or pixels as features is among else that words are more discrete and that if the feature 'fire truck' shows up on two different pictures they are likely to contain similar objects while two similar pixels might contain two completely different objects. This is one of the reasons to why it is easier to find patterns in words than in pixels.


# ## Part 4: Visualizing the learned features


first_layer_weights = vggmodel.get_weights()[0]
first_layer_weights.shape

def kernel_image(weights, i, positive):
    
    # extract the convolutional kernel at position i
    k = weights[:,:,:,i].copy()
    if not positive:
        k = -k
    
    # clip the values: if we're looking for positive
    # values, just keep the positive part; vice versa
    # for the negative values.
    k *= k > 0

    # rescale the colors, to make the images less dark
    m = k.max()
    if m > 1e-3:
        k /= m 

    return k

image_true = kernel_image(first_layer_weights, 10, positive=True)
image_false = kernel_image(first_layer_weights, 10, positive=False)

plt.imshow(image_true);

plt.imshow(image_false);

