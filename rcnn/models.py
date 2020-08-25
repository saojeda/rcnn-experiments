from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.layers import Conv1D, Conv2D
from tensorflow.keras.layers import Activation, BatchNormalization, Concatenate, add, Add
from tensorflow.keras.layers import GlobalAveragePooling1D, MaxPool1D, MaxPooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Model


class BaseModel:
    def __init__(self, ts_length, num_variables, loss, epochs, batch_size, optimizer):
        self.name = 'Base'
        self.model = None
        self.num_variables = num_variables
        self.ts_length = ts_length
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer

    def fit(self, dataset):
        X_train, Y_train = dataset['X_train'], dataset['Y_train']
        X_test, Y_test = dataset['X_test'], dataset['Y_test']

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=0.0001)
        early_stopping = EarlyStopping(monitor='val_loss', patience=200)

        # Train model
        self.model.fit(X_train, Y_train, epochs=self.epochs, batch_size=self.batch_size,
                       verbose=0, validation_split=0.2, callbacks=[reduce_lr, early_stopping])
        _, rmse = self.model.evaluate(X_test, Y_test)

        return rmse

    def build_model(self):
        pass


class CNNModel(BaseModel):
    def __init__(self, img_size, num_variables, loss, epochs, batch_size, optimizer):
        super().__init__(ts_length=None,
                         num_variables=num_variables,
                         loss=loss,
                         epochs=epochs,
                         batch_size=batch_size,
                         optimizer=optimizer)
        self.img_size = img_size
        self.name = 'CNN'
        self.model = self.build_model()

    def build_model(self):
        height, width, n_channels = self.img_size, self.img_size, self.num_variables

        model = Sequential([
            Conv2D(filters=32, kernel_size=3, padding='same', input_shape=(height, width, n_channels)),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=2),
            Conv2D(filters=32, kernel_size=3, padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=2),
            Dropout(0.25),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.25),
            Dense(self.num_variables, activation='linear')
        ])

        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=[RootMeanSquaredError()])

        return model


class FCNModel(BaseModel):
    def __init__(self, ts_length, num_variables, loss, epochs, batch_size, optimizer):
        super().__init__(ts_length=ts_length,
                         num_variables=num_variables,
                         loss=loss,
                         epochs=epochs,
                         batch_size=batch_size,
                         optimizer=optimizer)
        self.name = 'FCN'
        self.model = self.build_model()

    def build_model(self):
        input_layer = Input((self.ts_length, self.num_variables))

        conv1 = Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation(activation='relu')(conv1)

        conv2 = Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)

        conv3 = Conv1D(128, kernel_size=3, padding='same')(conv2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)

        gap_layer = GlobalAveragePooling1D()(conv3)

        output_layer = Dense(self.num_variables, activation='linear')(gap_layer)

        model = Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=[RootMeanSquaredError()])

        return model


class InceptionTimeModel(BaseModel):
    def __init__(self, ts_length, num_variables, loss, epochs, batch_size, optimizer):
        super().__init__(ts_length=ts_length,
                         num_variables=num_variables,
                         loss=loss,
                         epochs=epochs,
                         batch_size=batch_size,
                         optimizer=optimizer)
        self.name = 'InceptionTime'
        self.nb_filters = 32
        self.use_residual = True
        self.use_bottleneck = True
        self.depth = 6
        self.kernel_size = 40
        self.bottleneck_size = 32
        self.model = self.build_model()

    def _inception_module(self, input_tensor, stride=1, activation='linear'):
        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        max_pool_1 = MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = Concatenate(axis=2)(conv_list)
        x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = BatchNormalization()(shortcut_y)

        x = Add()([shortcut_y, out_tensor])
        x = Activation('relu')(x)
        return x

    def build_model(self):
        input_layer = Input((self.ts_length, self.num_variables))

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):
            x = self._inception_module(x)
            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = GlobalAveragePooling1D()(x)
        output_layer = Dense(self.num_variables, activation='linear')(gap_layer)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=[RootMeanSquaredError()])

        return model


class ResNetModel(BaseModel):
    def __init__(self, ts_length, num_variables, loss, epochs, batch_size, optimizer):
        super().__init__(ts_length=ts_length,
                         num_variables=num_variables,
                         loss=loss,
                         epochs=epochs,
                         batch_size=batch_size,
                         optimizer=optimizer)
        self.name = 'ResNet'
        self.model = self.build_model()

    def build_model(self):
        n_feature_maps = 64
        input_layer = Input((self.ts_length, self.num_variables))

        # BLOCK 1
        conv_x = Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
        conv_z = BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = BatchNormalization()(shortcut_y)

        output_block_1 = add([shortcut_y, conv_z])
        output_block_1 = Activation('relu')(output_block_1)

        # BLOCK 2
        conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = BatchNormalization()(shortcut_y)

        output_block_2 = add([shortcut_y, conv_z])
        output_block_2 = Activation('relu')(output_block_2)

        # BLOCK 3
        conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = BatchNormalization()(output_block_2)

        output_block_3 = add([shortcut_y, conv_z])
        output_block_3 = Activation('relu')(output_block_3)

        # FINAL
        gap_layer = GlobalAveragePooling1D()(output_block_3)
        output_layer = Dense(self.num_variables, activation='linear')(gap_layer)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=[RootMeanSquaredError()])

        return model