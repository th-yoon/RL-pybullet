import matplotlib.pyplot as plt

def save_plot(x, y, x_label, y_label, save_path):
    fig = plt.figure()
    plt.plot(x, y)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    fig.savefig(save_path)
    plt.close()

def save_variance_plot(x, y, x_label, y_label, variance, interval, save_path):
    fig = plt.figure()
    plt.errorbar(x, y, yerr=variance, ecolor='black')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    fig.savefig(save_path)
    plt.close()