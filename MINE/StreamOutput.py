import pandas as pd
import matplotlib.pyplot as plt

class StreamOutput:
    @staticmethod
    def display_plot(stream: pd.DataFrame, title: str = None, range: tuple[float, float] = None):
        """
        :param stream: The stream to be plotted. It must contain Value and Timestamp columns.
        :param title: If stated, this will be the title of the plot.
        :param range: If stated, this will plot only data within the specific index ranges.
        """

        if range is not None: plt.plot(stream["Timestamp"][range[0]:range[1]], stream["Value"][range[0]:range[1]])
        else : plt.plot(stream["Timestamp"], stream["Value"])
        if title is not None: plt.title(title)

        plt.title("Filtered Stream PPG")
        plt.show()
        plt.close()