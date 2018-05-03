"""
    Created by yuchen on 4/26/18
    Description:
"""
import numpy as np
import torch

def InfEightGaussiansGen(batch_size):
    """
    generate infinite 8 point real data
    :return: a batch size of these 8 points
    """
    centers = np.array([
        [1.414,0], [0,1.414], [-1.414,0], [0,-1.414],
        [1,1], [1,-1], [-1,-1], [-1,1]
    ])

    while True:
        ind = np.random.choice(8, batch_size, replace=True)
        data = np.random.randn(batch_size, 2)
        data = centers[ind] + data * 0.02
        yield data


def InfTwentyFiveGaussiansGen(batch_size):
    """
    generate infinite 25 point real data
    :return: a batch size of these 8 points
    """
    centers = np.array([
        [-2, -2], [-1, -2], [0, -2], [1, -2], [2, -2],
        [-2, -1], [-1, -1], [0, -1], [1, -1], [2, -1],
        [-2, 0], [-1, 0], [0, 0], [1, 0], [2, 0],
        [-2, 1], [-1, 1], [0, 1], [1, 1], [2, 1],
        [-2, 2], [-1, 2], [0, 2], [1, 2], [2, 2],
    ])

    while True:
        ind = np.random.choice(25, batch_size, replace=True)
        data = np.random.randn(batch_size, 2)
        data = centers[ind] + data * 0.02
        yield data


if __name__ == '__main__':
    # test code
    import matplotlib.pyplot as plt
    data_gen = InfEightGaussiansGen(batch_size=500)
    data_gen = InfTwentyFiveGaussiansGen(batch_size=500)
    data = data_gen.__next__()
    plt.plot(data[:,0], data[:,1], '.')
    plt.show()

