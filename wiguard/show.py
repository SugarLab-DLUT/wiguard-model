# python -m wiguard.show csidata.csv
import matplotlib.pyplot as plt
import sys
import pandas as pd

from wiguard.process import get_amplitude


def show(csi: pd.DataFrame):
    amplitude = get_amplitude(csi).T
    plt.imshow(amplitude, cmap='jet', interpolation='nearest', aspect='auto')
    plt.colorbar(label='Amplitude')
    plt.title('ESP CSI Amplitude Heatmap')
    plt.xlabel('Packet Index')
    plt.ylabel('Subcarrier Index')
    plt.show()


if __name__ == '__main__':
    csi = pd.read_csv(sys.argv[1], header=None)
    show(csi)
