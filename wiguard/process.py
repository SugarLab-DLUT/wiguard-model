import numpy as np
import pandas as pd


def get_complex(csidata: np.ndarray) -> np.ndarray:
    '''
    将 csidata 转换为复数数组

    @param csidata: 一维数组，每两个元素为一个复数的实部和虚部
    '''
    return np.vectorize(complex)(csidata[::2], csidata[1::2])


def get_amplitude(csi: pd.DataFrame) -> np.ndarray:
    '''
    从 csi 数据中提取幅度数据
    '''

    # csi 的最后一列为 csidata
    csidata = np.array([np.array(eval(data))
                       for data in csi.iloc[:, -1].values])
    csidata = np.vstack([get_complex(csi) for csi in csidata])
    amplitude = np.abs(csidata)
    return amplitude
