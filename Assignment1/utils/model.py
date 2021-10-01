from tensorflow.keras.layers import Dense, Flatten, Input
from keras.layers import Rescaling
from tensorflow.keras.models import Sequential
from pandas import DataFrame
import matplotlib.pyplot as plt
import os

class ClassifierModel:
    def __init__(self):
        self.model = None
        self.history = None

    def create_model(self):
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
        print(self.model.summary())
    
    def compile_model(self, loss="sparse_categorical_crossentropy", optimizer="SGD", metrics="accuracy"):
        try:
            self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        except:
            print("Build Model First")
    
    def train(self, X_train, y_train, epoch, batch=32, val_set=None):
        try:
            self.history = self.model.fit(X_train, y_train, epochs=epoch, batch_size=batch, validation_data=val_set)
        except:
            print("Build Model First")
    
    def evaluate(self, X_test, y_test):
        try:
            return self.model.evaluate(X_test, y_test)
        except:
            print("Build Model First")
    
    def plot_history(self):
        try:
            DataFrame(self.history.history).plot(figsize=(10,7))
            plt.grid(True)
            plt.show()
        except:
            print("Train model first")
    
    def save_model(self, filename):
        cwd = os.getcwd()
        path = os.path.join(cwd, "saved_model")
        os.makedirs(path, exist_ok=True)

        path = os.path.join(path, filename)
        self.model.save(path)

    