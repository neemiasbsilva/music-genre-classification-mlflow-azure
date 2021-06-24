import tensorflow as tf
import tensorflow.keras as keras


class BuildModel:

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.filters = [16, 32, 64]
        self.kernel = [(3, 5), (3, 1), (1, 5)]
        self.strides = [(1, 1), (1, 1), (1, 1)]
        self.max_pooling = [(2, 2), (2, 1), (2, 1)]


    def cnn_block(self, model, filters, kernel, strides, max_pooling):
        
        cnn_layer = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel,
            strides=strides,
            padding="same",
            kernel_regularizer=keras.regularizers.l2(0.0001)
        )(model)

        batch_norm = keras.layers.BatchNormalization(
            axis=-1,
            momentum=0.1,
            epsilon=1e-05,
        )(cnn_layer)

        cnn_layer = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel,
            strides=strides,
            padding="same",
            kernel_regularizer=keras.regularizers.l2(0.0001)
        )(batch_norm)


        batch_norm = keras.layers.BatchNormalization(
            axis=-1
        )(cnn_layer)


        activation = keras.layers.Activation(keras.activations.relu)(batch_norm)

        cnn_module = keras.layers.MaxPooling2D(pool_size=max_pooling, strides=max_pooling)(activation)
        
        return cnn_module


    def feed_foward(self):

        input = keras.layers.Input(shape=self.input_shape)

        for i in range(len(self.filters)):
            if (i == 0):
                x = self.cnn_block(input, filters=self.filters[i], kernel=self.kernel[i],
                                   strides=self.strides[i], max_pooling=self.max_pooling[i])
            else:
                x = self.cnn_block(x, filters=self.filters[i], kernel=self.kernel[i],
                                   strides=self.strides[i], max_pooling=self.max_pooling[i])

        global_avg_pool = keras.layers.GlobalAveragePooling2D(data_format="channels_last")(x)

        dense_layer = keras.layers.Dense(128, activation='relu')(global_avg_pool)

        predictions = keras.layers.Dense(self.output_shape,
                            activation='softmax')(dense_layer)

        model = keras.models.Model(inputs=input, outputs=predictions)

        return model
