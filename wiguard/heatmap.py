import matplotlib.pyplot as plt
from wiguard.dataset import LABELS, CSIDataset
import os

CUT_LEN = False

fig, ax = plt.subplots()

heatmap = ax.imshow([[0]], cmap='jet',
                    interpolation='nearest', aspect='auto', origin='lower')


def update(data):
    shape = data.shape
    heatmap.set_data(data[:, 1:].T)
    heatmap.set_clim(vmin=-2, vmax=2)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.draw()

if __name__ == '__main__':
    data_path = './data/241010'
    csi_dataset = CSIDataset(data_path)
    cnt = 0
    for data, label in csi_dataset:
        pic_path = os.path.join(data_path.replace('data', 'pic'), LABELS[label])
        if not os.path.exists(pic_path):
            os.makedirs(pic_path)
        update(data)
        plt.savefig(os.path.join(pic_path, f'{cnt}.png'))
        cnt += 1