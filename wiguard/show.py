import matplotlib.pyplot as plt
import sys
import pandas as pd
import numpy as np

from wiguard.dataset import CLIP_SIZE, SUBCARRIES
from wiguard.process import get_amplitude

fig, ax = plt.subplots()
plt.title('ESP CSI Amplitude Heatmap')
plt.xlabel('Packet Index')
plt.ylabel('Subcarrier Index')

heatmap = ax.imshow([[0]], cmap='jet',
                    interpolation='nearest', aspect='auto', origin='lower')
cbar = plt.colorbar(heatmap, ax=ax, label='Amplitude')
stats = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white')


def update(csi: pd.DataFrame):
    amplitude = get_amplitude(csi).T
    shape = amplitude.shape
    heatmap.set_data(amplitude)
    heatmap.set_extent((0, shape[1], 0, shape[0]))
    heatmap.set_clim(vmin=0, vmax=np.max(amplitude))
    ax.set_xlim(0, shape[1])
    ax.set_ylim(0, shape[0])
    stats.set_text(
        f"MIN: {np.min(amplitude):.2f} MAX: {np.max(amplitude):.2f} MEAN: {np.mean(amplitude):.2f}")
    plt.draw()


if __name__ == '__main__':
    csi = pd.read_csv(sys.argv[1], header=None)
    update(csi)
    plt.show()
