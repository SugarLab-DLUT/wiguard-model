import sys
from csiread import Intel
import numpy as np
import matplotlib.pyplot as plt

csidata = Intel(sys.argv[1])
csidata.read()

ampl = np.abs(csidata.get_scaled_csi())
rx = ampl.shape[2]
tx = ampl.shape[3]

plt.figure(f'amplitude of {sys.argv[1]}')

for i in range(rx):
    for j in range(tx):
        plt.subplot(rx, tx, i * tx + j + 1)
        plt.imshow(ampl[:, :, i, j].T, cmap='jet',
                   aspect='auto', vmin=0, vmax=36)
        plt.colorbar()
        plt.title(f"pair_{i}_{j}", fontsize=10)

plt.subplots_adjust(hspace=0.5)
plt.show()
