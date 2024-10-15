import matplotlib.pyplot as plt
import numpy as np

arrays = []

with open(f"sync_scheduler/log_stats_proj_2_0.txt","r") as fd:
        
        for ln in fd.readlines():
            if "LOSS" in ln:
                
                vl = float(ln.split("LOSS:")[1].strip())
                arrays.append(vl)

arrays = arrays[::10]

arrays = np.array(arrays)



arrays = np.convolve(arrays,[0.1 for _ in range(10)],"valid")

plt.plot(arrays,label="sync 75%")

plt.legend()
# plt.savefig("shuffle_w4p50.pdf")
plt.show()