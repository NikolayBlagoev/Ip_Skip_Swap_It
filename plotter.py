import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(8,4))
import os




def add_to_plot(dir_name, lbl = None):
    orig_name = dir_name
    dir_name = os.fsencode(dir_name)
    curr = []
    for file in os.listdir(dir_name):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"): 
            with open(f"{orig_name}/{filename}","r") as fd:
    
                for ln in fd.readlines():
                    try:
                        curr.append(float(ln))
                    except ValueError:
                        continue

            curr = np.array(curr)
            curr = np.convolve(curr,[0.1 for _ in range(10)],"valid")
            if lbl == None:
                splts = filename.split("_")
                if splts[2] == "0.txt":
                    lbl = f"baseline"
                else:
                    lbl = f"{splts[1]} {splts[2][:-4]}% faults"
            plt.plot(curr, label=lbl)
            vl = np.argmax(curr < 0.9)
            print(vl,lbl)
            return
        else:
            continue


if __name__ == '__main__':
    setting = "staggler"
    group = "tests_iid_fail"
    # add_to_plot('test_prev/4_full_length_paths',lbl="baseline")

    # add_to_plot('test_prev/4_full_length_paths',lbl="full model")
    # add_to_plot('test_prev/4_34_length_paths',lbl="random 75%")
    # add_to_plot('tests_1_fail/0',lbl="round robin 75%")
    # add_to_plot('tests_iid_fail/0_download', lbl="Baseline")
    # add_to_plot('tests_iid_fail/staggler', lbl="Staggler")
    # add_to_plot('tests_iid_fail/checkpoint', lbl="Checkpoint")
    # add_to_plot('tests_iid_fail/1_back', lbl="Back skip")
    # add_to_plot('tests_iid_fail/1_backdrop', lbl="Back drop")
    # add_to_plot(f"{group}/staggler",lbl = f"0.1% staggler")
    # add_to_plot(f"{group}/download",lbl = f"0.1% download")
    # add_to_plot(f"{group}/checkpoint",lbl = f"0.1% checkpoint")
    # 8000
    # add_to_plot(f"{group}/0_no_skip",lbl = f"baseline")
    # add_to_plot(f"{group}/0_skip",lbl = f"75% round robin")
    # add_to_plot(f"{group}/0_xor",lbl = f"75% xor")
    # add_to_plot(f"{group}/0_no_skip",lbl = f"baseline")
    # add_to_plot(f"{group}/0_no_skip",lbl = f"no skip")
    # add_to_plot(f"{group}/1_swap",lbl = f"1 layer swapped each microbatch")
    # add_to_plot(f"{group}/2_swap",lbl = f"2 layers swapped each microbatch")
    # add_to_plot(f"{group}/0_no_skip",lbl = f"100% model no faults")
    add_to_plot(f"{group}/3_back_drop",lbl = f"75% model round robin 3% faults (skip)")
    add_to_plot(f"{group}/3_back_skip",lbl = f"75% model round robin 3% faults (drop)")
    # add_to_plot(f"{group}/10_skip",lbl = f"75% model round robin 10% faults (skip)")
    # add_to_plot(f"{group}/3_skip_xor",lbl = f"75% model xor 3% faults (skip)")
    # add_to_plot(f"{group}/5_skip",lbl = f"75% model 5% faults (skip)")
    # add_to_plot(f"{group}/3_drop",lbl = f"3% drop")
    # add_to_plot(f"{group}/0_xor",lbl = f"75% xor")
    # add_to_plot(f"{group}/1_checkpoint",lbl = f"1% baseline failure")
    
    # add_to_plot(f"{group}/1_download",lbl = f"1% download failure")
    # add_to_plot(f"{group}/1_staggler",lbl = f"1% staggler failure")
    # add_to_plot(f"{group}/1_back",lbl = f"1% skip back"
    # add_to_plot(f"{group}/1_backdrop",lbl = f"1% drop back")
    # add_to_plot(f"{group}/checkpoint",lbl = f"0.1% checkpoint")
    # plt.xticks([i*800 for i in range(11)], [i*10*800 for i in range(11)])

    plt.ylabel("Validation Loss")
    plt.xlabel("Iteration")
    plt.legend(loc="upper right")
    plt.savefig(f"xorrr10.pdf")
    # plt.savefig(f"{setting}+{group}.pdf")
    plt.show()
