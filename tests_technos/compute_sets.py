import json
import numpy as np

f = open("all_database.json", "r")
database = json.load(f)
f.close()

n = len(database)

f_parameters_list = {}
s_parameters_list = {}

for i in range(len(database)):
    for param in database[i]["F_TRIZ_PARAMS"]:
        if param not in f_parameters_list:
            f_parameters_list[param] = np.zeros(n)
        f_parameters_list[param][i]=1
    for param in database[i]["S_TRIZ_PARAMS"]:
        if param not in s_parameters_list:
            s_parameters_list[param] = np.zeros(n)
        s_parameters_list[param][i]=1

np.save("f_parameters_list", f_parameters_list)
np.save("s_parameters_list", s_parameters_list)
