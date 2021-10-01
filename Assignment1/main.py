from utils.model import ClassifierModel
from utils.create_dataset import create_data

if __name__=='__main__':
    LOOP = True

    while LOOP:

        print("""
        =================================================

            Select the Dataset
            Press 1 --> MNIST
            Press 2 --> Fashion MNIST

        =================================================\n
        """)

        try:
            dataset = int(input("Enter Dataset: "))
        except ValueError:
            print("Please enter Number.")

        if dataset == 1:
            (X_train, y_train), val, (X_test, y_test) = create_data(1)
        elif dataset == 2:
            (X_train, y_train), val, (X_test, y_test) = create_data(2)
        else:
            print("Please Select 1 or 2")
            input("Press Enter to continue...")
            continue

        model = ClassifierModel()
        
        model.create_model()
        model.compile_model()

        try:
            EPOCH = int(input("Enter Epoch: "))
            BATCH = int(input("Enter Batch Size: "))
        except ValueError:
            print("Please Enter Number")
            input("Press Enter to continue...")
            continue

        model.train(X_train, y_train, EPOCH, BATCH, val)

        model.plot_history()

        model.evaluate(X_test, y_test)

        response = input("Save model y/n: ").lower()
        if response == "y":
            file_name = input("Enter File Name: ")
            if not file_name.endswith(".h5"):
                file_name += ".h5"
            model.save_model(file_name)
            print("Model saved Sucessfully!!")
        else:
            pass

        end_input = input("Press q to Quit....: ").lower()
        if end_input == "q":
            LOOP = False
        