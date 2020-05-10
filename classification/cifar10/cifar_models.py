"""
Separated File containing all different models implemented

Creation Date: May 2020
Creator: GranScudetto
"""
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Dense
from tensorflow.keras.layers import MaxPool2D, Concatenate, Flatten
from tensorflow.keras import Model


def model_2(input_shape, nb_classes):
    
    # 32, 16, 8, 4, 2
    inp = Input(shape=input_shape)  # 32 x 32
    
    conv_3x3_1 = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(inp)
    conv_3x3_1 = BatchNormalization()(conv_3x3_1)
    conv_3x3_1 = Activation(activation='relu')(conv_3x3_1)
    
    conv_5x5_1 = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(inp)
    conv_5x5_1 = BatchNormalization()(conv_5x5_1)
    conv_5x5_1 = Activation(activation='relu')(conv_5x5_1)
    
    network_layer_1 = Concatenate()([conv_3x3_1, conv_5x5_1])
    network_layer_1_pooled = MaxPool2D(pool_size=(2, 2))(network_layer_1)  # 16x16
    
    conv_3x3_2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(network_layer_1_pooled)
    conv_3x3_2 = BatchNormalization()(conv_3x3_2)
    conv_3x3_2 = Activation(activation='relu')(conv_3x3_2)
    
    conv_5x5_2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(network_layer_1_pooled)
    conv_5x5_2 = BatchNormalization()(conv_5x5_2)
    conv_5x5_2 = Activation(activation='relu')(conv_5x5_2)
    
    scaled_input = MaxPool2D(pool_size=(2, 2))(inp)
    conv_3x3_1_3 = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(scaled_input)
    conv_3x3_1_3 = BatchNormalization()(conv_3x3_1_3)
    conv_3x3_1_3 = Activation(activation='relu')(conv_3x3_1_3)
    conv_3x3_2_3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(conv_3x3_1_3)
    conv_3x3_2_3 = BatchNormalization()(conv_3x3_2_3)
    conv_3x3_2_3 = Activation(activation='relu')(conv_3x3_2_3)
    
    network_layer_2 = Concatenate()([conv_3x3_2, conv_5x5_2, conv_3x3_2_3])
    network_layer_2_pooled = MaxPool2D(pool_size=(2, 2))(network_layer_2)  # 8x8
    
    conv_3x3_3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(network_layer_2_pooled)
    conv_3x3_3 = BatchNormalization()(conv_3x3_3)
    conv_3x3_3 = Activation(activation='relu')(conv_3x3_3)
    
    conv_3x3_3_3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(conv_3x3_2_3)
    conv_3x3_3_3 = BatchNormalization()(conv_3x3_3_3)
    conv_3x3_3_3 = Activation(activation='relu')(conv_3x3_3_3)
    
    conv_3x3_3_3 = MaxPool2D(pool_size=(2, 2))(conv_3x3_3_3)
    network_layer_3 = Concatenate()([conv_3x3_3, conv_3x3_3_3])
    network_layer_3_pooled = MaxPool2D(pool_size=(2, 2))(network_layer_3)
    
    conv_3x3_4 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(network_layer_3_pooled)
    conv_3x3_4 = BatchNormalization()(conv_3x3_4)
    conv_3x3_4 = Activation(activation='relu')(conv_3x3_4)
    
    conv_3x3_5 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(conv_3x3_4)
    conv_3x3_5 = BatchNormalization()(conv_3x3_5)
    conv_3x3_5 = Activation(activation='relu')(conv_3x3_5)
    
    conv_3x3_6 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(conv_3x3_5)
    conv_3x3_6 = BatchNormalization()(conv_3x3_6)
    conv_3x3_6 = Activation(activation='relu')(conv_3x3_6)
    
    flattened = Flatten()(conv_3x3_6)
    flattened = Dense(units=128, activation='relu')(flattened)
    dense_pre_out = Dense(units=nb_classes, activation='relu')(flattened)
    
    out = Dense(units=nb_classes, activation='softmax')(dense_pre_out)
    
    return Model(inputs=inp, outputs=out)
