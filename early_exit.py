import os
stops = {4: [0,0,0,0,0],
        8: [0,0,0,0,0],
        12: [0,0,0,0,0],
        16: [0,0,0,0,0],
        20: [0,0,0,0,0],
        24: [0,0,0,0,0],
        28: [0,0,0,0,0],
        32: [0,0,0,0,0]}
with open("tests/test1.txt","r") as fd:
    tmp = {}
    curr = 0
    for ln in fd.readlines():
        if "STOP AT:" in ln:
            curr = int(ln.split(" ")[2])
        elif "indices" in ln:
            
            dt = ln.split("[[")[1].split("]]")[0].split(",")
            tmp[curr] = {}
            if curr == 32:
                tmp[curr]["best"] = [int(dt[0].strip())]
                # tmp[curr]["best"] = [int(el.strip()) for el in dt]
            else:
                tmp[curr]["best"] = int(dt[0].strip())
        elif "Norm" in ln:
            tmp[curr]["norm"] = float(ln.strip()[len("Norm tensor("):-1])
        elif "----------" in ln:
            if len(tmp) == 0:
                continue
            for i in range(1,8):
                agree = False
                if tmp[i*4]["best"] in tmp[curr]["best"]:
                    agree = True
                    stops[i*4][0] += 1
                    if "norm" in tmp[i*4]:
                        stops[i*4][1] += tmp[i*4]["norm"]
                else:
                    if "norm" in tmp[i*4]:
                        stops[i*4][2] += tmp[i*4]["norm"]
                agreed_before = 0
                count = 0
                for k in range(max(1,i-7),i):
                    count += 1
                    if  tmp[k*4]["best"] == tmp[i*4]["best"]:
                        agreed_before += 1
                if count == 0:
                    count = 1
                agreed_before = agreed_before/count
                if agree:
                    stops[i*4][3] += agreed_before
                else:
                    stops[i*4][4] += agreed_before
            tmp = {}
        continue
stops[4][1] = 0.001
for k,v in stops.items():
    print(k,v[0],v[1]/v[0], v[2]/(50-v[0]),v[3]/v[0],v[4]/(50-v[0]))