from tensorflow.keras.layers import Dense, Flatten, Input
from keras.layers import Rescaling
from tensorflow.keras.models import Sequential
from pandas import DataFrame
import matplotlib.pyplot as plt
import os

class ClassifierModel:
    r"""
    Class for classification model.
    """
    def __init__(self):
        self.model = None
        self.history = None

    def create_model(self):
        r"""
        Function is use to create model.

        :return: None
        """
        layers = [
            Input(shape=(28,28, 1), name="inputLayer"),
            Rescaling(scale=1./255),
            Flatten(name="flattenLayer"),

            Dense(300, activation="relu", name="hiddenLayer1"),
            Dense(100, activation="relu", name="hiddenLayer2"),
            Dense(300, activation="relu", name="hiddenLayer3"),

            Dense(10, activation="sigmoid", name="outputLayer")
        ]

        self.model = Sequential(layers=layers)
        print("\n", self.model.summary())
    
    def compile_model(self, loss="sparse_categorical_crossentropy", optimizer="SGD", metrics="accuracy"):
        r"""
        Function is use to compile model saved in self.model.

        :param loss: keras.loss, Loss function for model.
        :param optimizer: keras.optimizer, Optimizer for the model.
        :param metrics: keras.metrics, Mrtrics used for the model.

        :return: None
        """
        try:
            self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        except:
            print("Build Model First")
    
    def train(self, X_train, y_train, epoch, batch=32, val_set=None):
        r"""
        Function is use to train model saved in self.model.

        :param X_train: np.ndarray, X Training data.
        :param y_train: np.ndarray, y training data.
        :param epoch: int, Number of epochs to train the model.
        :param batch: int, Number of samples per gradient update.
        :param val_set: A tuple (x_val, y_val) of Numpy arrays, Data on which to evaluate the loss 
                        and any model metrics at the end of each epoch. 
        :return: None
        """
        try:
            self.history = self.model.fit(X_train, y_train, epochs=epoch, batch_size=batch, validation_data=val_set)
        except:
            print("Build Model First")
    
    def evaluate(self, X_test, y_test):
        r"""
        Function is use to evaluate model saved in self.model.

        :param X_train: np.ndarray, X Test data.
        :param y_train: np.ndarray, y Test data.

        :return: None
        """
        try:
            return self.model.evaluate(X_test, y_test)
        except:
            print("Build Model First")
    
    def plot_history(self):
        r"""
        Function is use to plot the training and validation loss and metric of every epoch

        :return: None
        """
        try:
            DataFrame(self.history.history).plot(figsize=(10,7))
            plt.grid(True)
            plt.show()
        except:
            print("Train model first")
    
    def save_model(self, filename):
        r"""
        Function is use to saved the model.

        :param filename: str, File name.

        :return: None
        """
        cwd = os.getcwd()
        path = os.path.join(cwd, "saved_model")
        os.makedirs(path, exist_ok=True)

        path = os.path.join(path, filename)
        self.model.save(path)

    