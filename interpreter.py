import matplotlib.pyplot as plt
import numpy as np
from plotter import add_to_plot
arrays = []

with open(f"async_scheduler/log_stats_proj_2_0.txt","r") as fd:
        tmp = []
        for ln in fd.readlines():
            
            if "LOSS" in ln:
                
                vl = float(ln.split("LOSS:")[1].strip())
                tmp.append(vl)
                if len(tmp) == 3:
                    arrays.append(sum(tmp)/len(tmp))
                    tmp = []
                

arrays = arrays[::10]

arrays = np.array(arrays)



arrays = np.convolve(arrays,[0.1 for _ in range(10)],"valid")
add_to_plot('tests_1_fail/0', lbl="Baseline")
plt.plot(arrays,label="async 75%")

plt.legend()
plt.savefig("async_scheduler.pdf")
plt.show()