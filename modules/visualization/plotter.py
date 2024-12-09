import matplotlib.pyplot as plt

def plot_signal(signal, title="Respiration Signal", xlabel="Time (frames)", ylabel="Amplitude"):
    """
    Plot the respiration signal.
    Args:
        signal (list or np.ndarray): Respiration signal.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(signal, label="Respiration Signal", color="blue")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()
