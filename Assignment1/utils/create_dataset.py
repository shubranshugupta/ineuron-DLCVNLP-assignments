from tensorflow.keras.datasets import mnist, fashion_mnist

def create_data(dataset, val_size=0.1):
    r"""
    Function is use to create dataset for the classification problem.

    :param dataset: it is use to select dataset
                    1. mnist dataset
                    2. fashion mnist dataset
    :param val_size: Float, Specify the size of validation data.
    
    :return: train, val, test dataset
    """
    if dataset == 1:
        (X, y), test = mnist.load_data()
    else:
        (X, y), test = fashion_mnist.load_data()

    len_val = int(X.shape[0] * val_size)
    
    return (X[len_val:], y[len_val:]), (X[:len_val], y[:len_val]), test


