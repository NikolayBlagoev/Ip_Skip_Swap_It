import json
import matplotlib.pyplot as plt
import numpy as np

with open("norms75.json","r") as fd:
    dc75 = json.load(fd)
with open("norms100.json","r") as fd:
    dc100 = json.load(fd)

for i in range(16):
    arr1  = np.array(dc75[str(i)])
    arr1 = np.convolve(arr1,[0.1 for _ in range(10)],"valid")
    arr2  = np.array(dc100[str(i)])
    arr2 = np.convolve(arr2,[0.1 for _ in range(10)],"valid")
    plt.title("LAYER "+str(i))
    plt.plot(arr1,label="75%")
    plt.plot(arr2, label="100%")
    plt.legend()
    plt.show()