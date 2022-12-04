import matplotlib.pyplot as plt

def model_perf_analysis(train_history, val_history, type_name, prefix):
    plt.figure()
    plt.title(prefix + " Training Summary of " + type_name)
    plt.plot(train_history, label='Training ' + type_name)
    plt.plot(val_history, label='Validation ' + type_name)
    plt.xlabel('Epochs')
    plt.ylabel(type_name)
    plt.legend()
    plt.show()