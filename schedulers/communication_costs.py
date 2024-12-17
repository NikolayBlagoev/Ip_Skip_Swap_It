
DELAY_BANDWIDTHS = {
    "Oregon-Virginia": (67, 0.395),
    "Oregon-Ohio": (49, 0.55),
    "Oregon-Tokyo": (96, 0.261),
    "Oregon-Seoul": (124, 0.23),
    "Oregon-Singapore": (163, 0.17),
    "Oregon-Sydney": (139, 0.18),
    "Oregon-London": (136, 0.21),
    "Oregon-Frankfurt": (143, 0.202),
    "Oregon-Ireland": (124, 0.241),
    "Virginia-Ohio": (11, 0.56),
    "Virginia-Tokyo": (143, 0.262),
    "Virginia-Seoul": (172, 0.250),
    "Virginia-Singapore": (230, 0.182),
    "Virginia-Sydney": (197, 0.191),
    "Virginia-London": (76, 0.58),
    "Virginia-Frankfurt": (90, 0.51),
    "Virginia-Ireland": (67, 0.55),
    "Ohio-Tokyo": (130, 0.347),
    "Ohio-Seoul": (159, 0.264),
    "Ohio-Singapore": (197, 0.226),
    "Ohio-Sydney": (185, 0.242),
    "Ohio-London": (86, 0.52),
    "Ohio-Frankfurt": (99, 0.400),
    "Ohio-Ireland": (77, 0.57),
    "Tokyo-Seoul": (34, 0.55),
    "Tokyo-Singapore": (73, 0.505),
    "Tokyo-Sydney": (100, 0.380),
    "Tokyo-London": (210, 0.183),
    "Tokyo-Frankfurt": (223, 0.18),
    "Tokyo-Ireland": (199, 0.232),
    "Seoul-Singapore": (74, 0.57),
    "Seoul-Sydney": (148, 0.29),
    "Seoul-London": (238, 0.171),
    "Seoul-Frankfurt": (235, 0.179),
    "Seoul-Ireland": (228, 0.167),
    "Singapore-Sydney": (92, 0.408),
    "Singapore-London": (169, 0.25),
    "Singapore-Frankfurt": (155, 0.219),
    "Singapore-Ireland": (179, 0.246),
    "Sydney-London": (262, 0.163),
    "Sydney-Frankfurt": (265, 0.164),
    "Sydney-Ireland": (254, 0.172),
    "London-Frankfurt": (14, 0.57),
    "London-Ireland": (12, 0.54),
    "Frankfurt-Ireland": (24, 0.54),
    "Sofia-Frankfurt":(25.05,0.160),
    "Sofia-Seoul":(323.03,0.128),
    "Sofia-Sydney":(298.90,0.134),
    "Sofia-Virginia":(120.46,0.143),
    "Sofia-Oregon":(114.24,0.096),
    "Sofia-Ohio":(125.96,0.097),
    "Sofia-London":(68.07,0.150),
    "Sofia-Tokyo":(318.88,0.130),
    "Sofia-Amsterdam":(38.24,0.150),
    "Amsterdam-Frankfurt":(8.72,0.150),
    "Amsterdam-Seoul":(288.39,0.134),
    "Amsterdam-Sydney":(265.94,0.135),
    "Amsterdam-Virginia":(80.81,0.150),
    "Amsterdam-Oregon":(72.29,0.150),
    "Amsterdam-Ohio":(75.31, 0.150),
    "Amsterdam-London":(7,0.150),
    "Amsterdam-Tokyo":(278.65,0.13),
}
COMPUTATIONAL_COST = {
    "Sofia": 5,
    'Singapore': 4.5, 
    'Ireland': 4.2,
    'Sydney': 3.6,
    'Frankfurt': 4.6, 
    'Seoul': 5.25, 
    'Oregon': 4.4, 
    'Frankfurt': 4.6, 
    'Ohio': 4, 
    'Tokyo': 4.9, 
    'Amsterdam': 3.9, 
    'Virginia': 4.1

}
def get_locations(setting = "geo-distributed"):
    if setting == "geo-distributed":
        return list(COMPUTATIONAL_COST.keys())
    elif setting == "single-cluster":
        return ["Amsterdam"]
    elif setting == "5-clusters":
        return ["Amsterdam", "Seoul", "Frankfurt", "Sydney", "Ohio"]
def id_to_loc(idx, setting="geo-distributed"):
    locs = get_locations(setting)
    idx = idx%len(locs)
    return locs[idx]
def get_computations(setting = "geo-distributed"):
    if setting == "geo-distributed":
        return COMPUTATIONAL_COST
    elif setting == "single-cluster":
        return {"Amsterdam": 2.0}
    elif setting == "5-clusters":
        ret = {}
        for loc in get_locations(setting):
            ret[loc] = 2.0
        return ret

def delay_map(loc1,loc2, sz = 250*6291908):
    p1 = loc1
    p2 = loc2
    if DELAY_BANDWIDTHS.get(p1+"-"+p2) != None:
        ret = DELAY_BANDWIDTHS.get(p1+"-"+p2)
    elif DELAY_BANDWIDTHS.get(p2+"-"+p1) != None:
        ret = DELAY_BANDWIDTHS.get(p2+"-"+p1)
    else:
        ret = (1,100)
    return ret[0]/1000 + sz/(1024**3 * ret[1])