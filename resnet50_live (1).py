import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.preprocessing.image import ImageDataGenerator

###################################################################
# Input parameters
###################################################################
channels = 3
num_classes = 15 #2 classes for cats v dogs
image_size = (224,224)
batch_size = 32


###################################################################
# Dataset - Cats v Dogs
###################################################################

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

# Creating training and testing sets
training_set = train_datagen.flow_from_directory('celebdataset/train',
                                                 target_size = image_size,
                                                 batch_size = batch_size,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('celebdataset/test',
                                            target_size = image_size,
                                            batch_size = batch_size,
                                            class_mode = 'categorical')


#################Build of major subbloc 0
#Input shape is 56,56,64


####################### START OF THE CONV RESNET BLOCK ################

#Kernels are 64K size 1x1
#then 64 of 3x3
#then 256 of a 1x1
block0__input = keras.Input(shape=(56,56,64),name='block0_input')
x = keras.layers.Conv2D(64, kernel_size=1, activation='relu', padding='same', strides=1, name='b0_c0')(block0__input)
x = keras.layers.BatchNormalization() (x)

#64x3x3
x = keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same', strides=1, name='b0_c1')(x)
x = keras.layers.BatchNormalization() (x)

#256x1x1
x = keras.layers.Conv2D(256, kernel_size=1, activation='relu', padding='same', strides=1, name='b0_c2')(x)
x = keras.layers.BatchNormalization() (x)

#Fast forward Resent layer now
###?? What is the input to the resenet layer
block0_transform = keras.layers.Conv2D(256, kernel_size=1, strides=1,activation='relu', padding='same')(block0__input)
x1 = keras.layers.BatchNormalization() (block0_transform)

#combine the Resnet FF sequence to the Conv Sequence
x = keras.layers.Add()([x,x1]) #add up the values from both the resnet path and regular path

#after adding things up, just do a RELU of the entire thing
block0_output = keras.layers.ReLU()(x)

#create a variable for the MODEL of this entire complex thing
block0_conv = keras.Model(inputs=block0__input, outputs = block0_output, name='block0_conv')

####################### END OF THE CONV RESNET BLOCK ################


block0__identity_input = keras.Input(shape=(56,56,256),name='block0_identity_input')
#Names are optional only if you need to reference specific layers
x = keras.layers.Conv2D(64, kernel_size=1, activation='relu', padding='same', strides=1)(block0__identity_input)
x = keras.layers.BatchNormalization() (x)

#64x3x3
x = keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same', strides=1)(x)
x = keras.layers.BatchNormalization() (x)

#256x1x1
x = keras.layers.Conv2D(256, kernel_size=1, activation='relu', padding='same', strides=1)(x)
x = keras.layers.BatchNormalization() (x)

#combine the Resnet FF sequence to the Conv Sequence
x = keras.layers.Add()([x,block0__identity_input]) #add up the values from both the resnet path and regular path

#after adding things up, just do a RELU of the entire thing
block0_identity_output = keras.layers.ReLU()(x)

#create a variable for the MODEL of this entire complex thing
block0_id = keras.Model(inputs=block0__identity_input, outputs = block0_identity_output, name='block0_id')

#### Finally, lets build a model that holds the Block0 Conv Resnet AND two Block0 ID Resnets

major_block0_in = keras.Input(shape=(56,56,64),name='major_block0_in')
# Per diagram, one RESNET CONV layer, followed by two ID layers
x = block0_conv(major_block0_in)
x = block0_id(x)
x = block0_id(x)

#Build another model to capture this sequence of models
block0= keras.Model(inputs=major_block0_in, outputs = x, name='block0')



#############################################################
#################end of block 0
#############################################################



###############################################
#BEGINNNING OF MAJOR BLOCK 1
###############################################
####################### START OF THE CONV RESNET BLOCK ################

#128
#128
#512
# one pool to reduce dimensions

block1_input = keras.Input(shape=(56,56,256),name='block1_input')
x = keras.layers.Conv2D(128, kernel_size=1, activation='relu', padding='same', strides=2)(block1_input)
x = keras.layers.BatchNormalization() (x)

#128x3x3
x = keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same', strides=1)(x)
x = keras.layers.BatchNormalization() (x)

#256x1x1
x = keras.layers.Conv2D(512, kernel_size=1, activation='relu', padding='same', strides=1)(x)
x = keras.layers.BatchNormalization() (x)

#Fast forward Resent layer now
#ALSO NEED STRIDES =2, and CONV changes Z dimention
#This conv should output a shape of 28x28x512 tensor
block1_transform = keras.layers.Conv2D(512, kernel_size=1, strides=2,activation='relu', padding='same')(block1_input)
x1 = keras.layers.BatchNormalization() (block1_transform)

#combine the Resnet FF sequence to the Conv Sequence
x = keras.layers.Add()([x,x1]) #add up the values from both the resnet path and regular path

#after adding things up, just do a RELU of the entire thing
block1_output = keras.layers.ReLU()(x)

#create a variable for the MODEL of this entire complex thing
block1_conv = keras.Model(inputs=block1_input, outputs = block1_output, name='block1_conv')

####################### END OF THE CONV RESNET BLOCK ################

########### beginning of block 1 identity connections/layers


block1_identity_input = keras.Input(shape=(28,28,512),name='block1_identity_input')
#Names are optional only if you need to reference specific layers
x = keras.layers.Conv2D(128, kernel_size=1, activation='relu', padding='same', strides=1)(block1_identity_input)
x = keras.layers.BatchNormalization() (x)

#64x3x3
x = keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same', strides=1)(x)
x = keras.layers.BatchNormalization() (x)

#256x1x1
x = keras.layers.Conv2D(512, kernel_size=1, activation='relu', padding='same', strides=1)(x)
x = keras.layers.BatchNormalization() (x)

#combine the Resnet FF sequence to the Conv Sequence
x = keras.layers.Add()([x,block1_identity_input]) #add up the values from both the resnet path and regular path

#after adding things up, just do a RELU of the entire thing
block1_identity_output = keras.layers.ReLU()(x)

#create a variable for the MODEL of this entire complex thing
block1_id = keras.Model(inputs=block1_identity_input, outputs = block1_identity_output, name='block1_id')


#### Finally, lets build a model that holds the Block0 Conv Resnet AND two Block0 ID Resnets

major_block1_in = keras.Input(shape=(56,56,256),name='major_block1_in')
# Per diagram, one RESNET CONV layer, followed by two ID layers
x = block1_conv(major_block1_in)
x = block1_id(x)
x = block1_id(x)

#Build another model to capture this sequence of models
block1 = keras.Model(inputs=major_block1_in, outputs = x, name='block1')






###############################################
#END OF MAJOR BLOCK 1
###############################################





def build_resnet_model():
    #define the input/input shape to the model,
    #use structural connectivity to build the model, NOT sequential

    resnet_input = keras.Input(shape=(image_size[0],image_size[1],channels),name='input')

    x = keras.layers.Conv2D(64, kernel_size=7, activation='relu', padding='same', strides=2, name='conv7x7')(resnet_input)

    x = keras.layers.MaxPooling2D(padding='same', strides = 2)(x)

    x = block0(x)
    x = block1(x)

    #Add the pooling layer
    x = keras.layers.MaxPooling2D(padding='same', strides = 2)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(10, activation='relu')(x)

    #output layer
    x = keras.layers.Dense(num_classes, activation='softmax')(x)


    #Lets build the model
    resnet_model= keras.Model(inputs=resnet_input, outputs = x, name='resnet')

    print(resnet_model.summary())

    return resnet_model
    # x = block0(x)
    # Build that major subblock0


#Call the build resnet function
model = build_resnet_model()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


#??? let go for broke and try and fit it

model.fit(training_set, epochs=10)


