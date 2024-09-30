# python -m wiguard.show csidata.csv
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd


def to_complex(arr):
    return np.vectorize(complex)(arr[::2], arr[1::2])


data = pd.read_csv(sys.argv[1], header=None)
csidata = np.array([np.array(eval(csi)) for csi in data.iloc[:, -1].values])
amplitude = np.abs(csidata)
print(amplitude)

csidata = np.vstack([to_complex(csi) for csi in csidata]).T


amplitude = np.abs(csidata)
print(amplitude)
plt.imshow(amplitude, cmap='jet', interpolation='nearest', aspect='auto')
plt.colorbar(label='Amplitude')
plt.title('ESP CSI Amplitude Heatmap')
plt.xlabel('Packet Index')
plt.ylabel('Subcarrier Index')
plt.show()
