import matplotlib.pyplot as plt
# %matplotlib inline

def model_accuracy(model_history):
    plt.plot(model_history.history['acc'])
    plt.plot(model_history.history['val_acc'])
    plt.title("Model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(['train', 'test'], loc="upper left")
    plt.show()
def model_loss(model_history):
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title("Model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(['train', 'test'], loc="upper left")
    plt.show()