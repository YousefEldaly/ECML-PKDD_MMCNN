import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, ZeroPadding1D, AveragePooling1D,
    Dense, Activation, Flatten, Input, Lambda, Reshape,
    GlobalAveragePooling1D, Multiply, Add, Concatenate, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers, metrics, losses, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, f1_score

class MMCNN_model():
    
    def __init__(self,
                channels=3,
                samples=1000):
        self.channels = channels
        self.samples = samples
        self.activation = 'elu'
        self.learning_rate = 0.0001
        self.dropout = 0.8
        self.inception_filters = [16, 16, 16, 16]
        # the parameter of the first part: EEG Inception block
        self.inception_kernel_length = [
            [5, 10, 15, 10],
            [40, 45, 50, 100],
            [60, 65, 70, 100],
            [80, 85, 90, 100],
            [160, 180, 200, 180],
        ]
        self.inception_stride = [2, 4, 4, 4, 16]
        self.first_maxpooling_size = 4
        self.first_maxpooling_stride = 4
        # the parameter of the second part: Residual block
        self.res_block_filters = [16, 16, 16]
        self.res_block_kernel_stride = [8, 7, 7, 7, 6]
        # the parameter of the third part: SE block
        self.se_block_kernel_stride = 16
        self.se_ratio = 8
        self.second_maxpooling_size = [4, 3, 3, 3, 2]
        self.second_maxpooling_stride = [4, 3, 3, 3, 2]
        
        self.model = self.build_model(self.channels, self.samples)
        adam = optimizers.Adam(learning_rate=self.learning_rate) 
        self.model.compile(loss=losses.binary_crossentropy,
                          optimizer=adam,
                          metrics=['mae', metrics.binary_accuracy])
            
    def build_model(self, channels, samples):
        output_conns = []
        input_tensor = Input(shape=(samples, channels))  
                
        '''
        EIN-a
        '''
        x = self.inception_block(input_tensor,
                                self.inception_filters,
                                self.inception_kernel_length[0],
                                self.inception_stride[0],
                                self.activation)

        x = MaxPooling1D(pool_size=self.first_maxpooling_size,
                         strides=self.first_maxpooling_stride,
                         padding='same')(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        x = self.conv_block(x,
                           self.res_block_filters,
                           self.res_block_kernel_stride[0],
                           activation=self.activation)
        
        x = self.squeeze_excitation_layer(x,
                                         self.se_block_kernel_stride,
                                         self.activation,
                                         ratio=self.se_ratio)
        x = MaxPooling1D(pool_size=self.second_maxpooling_size[0],
                         strides=self.second_maxpooling_stride[0],
                         padding='same')(x)
        x = Flatten()(x)
        output_conns.append(x)

        '''
        EIN-b
        '''
        y1 = self.inception_block(input_tensor,
                                 self.inception_filters,
                                 self.inception_kernel_length[1],
                                 self.inception_stride[1],
                                 self.activation)
        y1 = MaxPooling1D(pool_size=self.first_maxpooling_size,
                          strides=self.first_maxpooling_stride,
                          padding='same')(y1)
        y1 = BatchNormalization()(y1)
        y1 = tf.keras.layers.Dropout(self.dropout)(y1)
        y1 = self.conv_block(y1,
                            self.res_block_filters,
                            self.res_block_kernel_stride[1],
                            self.activation)
        y1 = self.squeeze_excitation_layer(y1,
                                          self.se_block_kernel_stride,
                                          self.activation,
                                          ratio=self.se_ratio)
        y1 = MaxPooling1D(pool_size=self.second_maxpooling_size[1],
                          strides=self.second_maxpooling_stride[1],
                          padding='same')(y1)
        y1 = Flatten()(y1)
        output_conns.append(y1)

        '''
        EIN-c
        '''
        y2 = self.inception_block(input_tensor,
                                 self.inception_filters,
                                 self.inception_kernel_length[2],
                                 self.inception_stride[2],
                                 self.activation)
        y2 = MaxPooling1D(pool_size=self.first_maxpooling_size,
                          strides=self.first_maxpooling_stride,
                          padding='same')(y2)
        y2 = BatchNormalization()(y2)
        y2 = tf.keras.layers.Dropout(self.dropout)(y2)
        y2 = self.conv_block(y2,
                            self.res_block_filters,
                            self.res_block_kernel_stride[2],
                            self.activation)
        y2 = self.squeeze_excitation_layer(y2,
                                          self.se_block_kernel_stride,
                                          self.activation,
                                          ratio=self.se_ratio)
        y2 = MaxPooling1D(pool_size=self.second_maxpooling_size[2],
                          strides=self.second_maxpooling_stride[2],
                          padding='same')(y2)
        y2 = Flatten()(y2)
        output_conns.append(y2)

        '''
        EIN-d
        '''
        y3 = self.inception_block(input_tensor,
                                 self.inception_filters,
                                 self.inception_kernel_length[3],
                                 self.inception_stride[3],
                                 self.activation)
        y3 = MaxPooling1D(pool_size=self.first_maxpooling_size,
                          strides=self.first_maxpooling_stride,
                          padding='same')(y3)
        y3 = BatchNormalization()(y3)
        y3 = tf.keras.layers.Dropout(self.dropout)(y3) 
        y3 = self.conv_block(y3,
                            self.res_block_filters,
                            self.res_block_kernel_stride[3],
                            self.activation)
        y3 = self.squeeze_excitation_layer(y3,
                                          self.se_block_kernel_stride,
                                          self.activation,
                                          ratio=self.se_ratio)
        y3 = MaxPooling1D(pool_size=self.second_maxpooling_size[3],
                          strides=self.second_maxpooling_stride[3],
                          padding='same')(y3)
        y3 = Flatten()(y3)
        output_conns.append(y3)

        '''
        EIN-e
        '''
        z = self.inception_block(input_tensor,
                                self.inception_filters,
                                self.inception_kernel_length[4],
                                self.inception_stride[4],
                                self.activation)
        z = MaxPooling1D(pool_size=self.first_maxpooling_size,
                         strides=self.first_maxpooling_stride,
                         padding='same')(z)
        z = BatchNormalization()(z)
        z = tf.keras.layers.Dropout(self.dropout)(z)
        z = self.conv_block(z,
                           self.res_block_filters,
                           self.res_block_kernel_stride[4],
                           self.activation)
        z = self.squeeze_excitation_layer(z,
                                         self.se_block_kernel_stride,
                                         self.activation,
                                         ratio=self.se_ratio)
        z = MaxPooling1D(pool_size=self.second_maxpooling_size[4],
                         strides=self.second_maxpooling_stride[4],
                         padding='same')(z)
        z = Flatten()(z)
        output_conns.append(z)

        output_conns = Concatenate(axis=-1)(output_conns)
        output_conns = tf.keras.layers.Dropout(self.dropout)(output_conns)
        output_tensor = Dense(2, activation='sigmoid')(output_conns)
        model = Model(input_tensor, output_tensor)
        return model
        
    '''
    se_block
    '''
    def squeeze_excitation_layer(self, x, out_dim, activation, ratio=8):
        squeeze = GlobalAveragePooling1D()(x)

        excitation = Dense(units=out_dim//ratio)(squeeze)
        excitation = Activation(activation)(excitation)
        excitation = Dense(units=out_dim, activation='sigmoid')(excitation)
        excitation = Reshape((1, out_dim))(excitation)

        scale = Multiply()([x, excitation])
        return scale
        
    '''
    res_block
    '''
    def conv_block(self, x, nb_filter, length, activation):
        k1, k2, k3 = nb_filter

        out = Conv1D(k1, length, strides=1, padding='same', 
                    kernel_regularizer=regularizers.l2(0.002))(x)
        out = BatchNormalization()(out)
        out = Activation(activation)(out)

        out = Conv1D(k2, length, strides=1, padding='same', 
                    kernel_regularizer=regularizers.l2(0.002))(out)
        out = BatchNormalization()(out)
        out = Activation(activation)(out)

        out = Conv1D(k3, length, strides=1, padding='same', 
                    kernel_regularizer=regularizers.l2(0.002))(out)
        out = BatchNormalization()(out)

        x = Conv1D(k3, 1, strides=1, padding='same')(x)
        x = BatchNormalization()(x)

        out = Add()([out, x])
        out = Activation(activation)(out)
        out = tf.keras.layers.Dropout(self.dropout)(out)
        return out
    
    '''
    inception_block 
    '''
    def inception_block(self, x, ince_filter, ince_length, stride, activation):
        k1, k2, k3, k4 = ince_filter
        l1, l2, l3, l4 = ince_length
        inception = []

        x1 = Conv1D(k1, l1, strides=stride, padding='same', 
                   kernel_regularizer=regularizers.l2(0.01))(x)
        x1 = BatchNormalization()(x1)
        x1 = Activation(activation)(x1)
        inception.append(x1)

        x2 = Conv1D(k2, l2, strides=stride, padding='same', 
                   kernel_regularizer=regularizers.l2(0.01))(x)
        x2 = BatchNormalization()(x2)
        x2 = Activation(activation)(x2)
        inception.append(x2)

        x3 = Conv1D(k3, l3, strides=stride, padding='same', 
                   kernel_regularizer=regularizers.l2(0.01))(x)
        x3 = BatchNormalization()(x3)
        x3 = Activation(activation)(x3)
        inception.append(x3)

        x4 = MaxPooling1D(pool_size=l4, strides=stride, padding='same')(x)
        x4 = Conv1D(k4, 1, strides=1, padding='same')(x4)
        x4 = BatchNormalization()(x4)
        x4 = Activation(activation)(x4)
        inception.append(x4)
        
        v1 = Concatenate(axis=-1)(inception)
        return v1
