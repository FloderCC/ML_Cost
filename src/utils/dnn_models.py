import keras
import numpy as np
from keras.src.layers import Flatten, Dropout
from keras.src.optimizers import Adam
from scikeras.wrappers import KerasClassifier
from keras.layers import Dense
from keras.models import Sequential
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')
keras.utils.disable_interactive_logging()

# Base Classifier
class DNNClassifier(KerasClassifier):
    __enabled_codified_predict = True

    def predict(self, X, **kwargs):
        y_pred = super().predict(X, **kwargs)
        if self.__enabled_codified_predict:
            y_pred = np.argmax(y_pred, axis=1)
        return y_pred

    def predict_proba(self, X, **kwargs):
        y_proba = super().predict(X, **kwargs)
        return y_proba

    def set_enabled_codified_predict(self, enabled):
        self.__enabled_codified_predict = enabled


# DNN model 1
# Paper: https://www.researchgate.net/profile/Madhusanka-Liyanage/publication/372250269_From_Opacity_to_Clarity_Leveraging_XAI_for_Robust_Network_Traffic_Classification/links/64acf0aac41fb852dd67fa41/From-Opacity-to-Clarity-Leveraging-XAI-for-Robust-Network-Traffic-Classification.pdf
# Source: https://github.com/Montimage/activity-classification/blob/master/xai/neural_networks_xai.ipynb
# Dataset: UNAC
class DNNClassifier1(DNNClassifier):
    def build(self, num_features, num_classes):
        # Creating a Keras Model
        model = Sequential()
        model.add(Dense(12, input_shape=(num_features,), activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(num_classes, activation='sigmoid'))
        # Compile the Keras model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # creating the model
        super().__init__(build_fn=model)

    def fit(self, X, y, **kwargs):
        # Add the desired parameters for fit
        kwargs['epochs'] = 150
        kwargs['batch_size'] = 10

        return super().fit(X, y, **kwargs)


# DNN model 2
# Paper: //Rafael's code
# Source: https://github.com/rgtzths/XAI_analysis/blob/main/Slicing5G/Slicing5G.py
# Dataset: Slicing5G
class DNNClassifier2(DNNClassifier):
    def build(self, num_features, num_classes):
        # Creating a Keras Model
        model = Sequential()
        model.add(Flatten(input_shape=(num_features,)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(num_classes, activation='softmax'))
        # Compile the Keras model
        # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.00001), metrics=['accuracy'])

        # creating the model
        super().__init__(build_fn=model)

    def fit(self, X, y, **kwargs):
        # Add the desired parameters for fit
        kwargs['epochs'] = 200
        kwargs['batch_size'] = 1024

        return super().fit(X, y, **kwargs)


# DNN model 3
# Paper: //Rafael's code
# Source: https://github.com/rgtzths/XAI_analysis/blob/main/IOT_DNL/IOT_DNL.py
# Dataset: IoT_DNL
class DNNClassifier3(DNNClassifier):
    def build(self, num_features, num_classes):
        # Creating a Keras Model
        model = Sequential()
        model.add(Flatten(input_shape=(num_features,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        # Compile the Keras model
        # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.000005), metrics=['accuracy'])

        # creating the model
        super().__init__(build_fn=model)

    def fit(self, X, y, **kwargs):
        # Add the desired parameters for fit
        kwargs['epochs'] = 200
        kwargs['batch_size'] = 1024

        return super().fit(X, y, **kwargs)


# DNN model 4
# Paper: https://ieeexplore.ieee.org/document/8993066
# Source: https://github.com/adtmv7/DeepSlice/blob/master/Source/DeepSlice.py
# Dataset: DeepSlice
class DNNClassifier4(DNNClassifier):
    def build(self, num_features, num_classes):
        # Creating a Keras Model
        model = Sequential()
        # model.add(Flatten(input_shape=(num_features,)))
        model.add(Dense(8, input_dim=num_features, kernel_initializer='normal', activation='relu'))
        model.add(Dense(4, kernel_initializer='normal', activation='relu'))
        model.add(Dense(3, kernel_initializer='normal', activation='tanh'))
        model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
        # Compile the Keras model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # creating the model
        super().__init__(build_fn=model)

    def fit(self, X, y, **kwargs):
        # Add the desired parameters for fit
        kwargs['epochs'] = 16
        kwargs['batch_size'] = 128

        return super().fit(X, y, **kwargs)
