from tensorflow.keras.datasets import mnist, fashion_mnist

def create_data(dataset, val_size=0.1):
    if dataset == 1:
        (X, y), test = mnist.load_data()
    else:
        (X, y), test = fashion_mnist.load_data()

    len_val = int(X.shape[0] * val_size)
    
    return (X[len_val:], y[len_val:]), (X[:len_val], y[:len_val]), test


