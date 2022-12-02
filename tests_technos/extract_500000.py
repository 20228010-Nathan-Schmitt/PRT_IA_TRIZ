import requests
import numpy as np
import json
from datetime import datetime, timedelta

def download_week(early, late, start=0):
    global database, f_parameters_list, s_parameters_list
    request_body = {
        "from": start,
        "size": 1000,
        "query": {
            "bool": {
                "must": [
                    {
                        "range": {
                            "PUBLICATION_DATE": {
                                "gte": early.strftime("%Y-%m-%d"),
                                "lt": late.strftime("%Y-%m-%d")
                            }
                        }
                    }

                ]
            }
        },
        "sort": [
            {
                "_id": {
                    "order": "asc"
                }
            }
        ],
        "fields": [
            "CONTRADICTION_SCORE",
            "F_SENTS",
            "S_SENTS",
            "F_TRIZ_PARAMS",
            "S_TRIZ_PARAMS"
        ],
        "_source": False
    }

    r = requests.get("https://vm-csip-es.icube.unistra.fr/db/db_solve/patents/_search", headers={"Authorization": "ApiKey ekRWY3ZINEI1b1ktTzQzX3ZhRGM6aTRlbVJjQXZUdzY2a3hDTmFmMVhoZw=="}, json=request_body, verify=False)
    response = r.json()


    for patent in response["hits"]["hits"]:
        database.append({
            "id": patent["_id"],
            "contradiction": patent["fields"]["F_SENTS"][0] + " " + patent["fields"]["S_SENTS"][0],  # contradiction
            "F_TRIZ_PARAMS": patent["fields"]["F_TRIZ_PARAMS"],
            "S_TRIZ_PARAMS": patent["fields"]["S_TRIZ_PARAMS"],
        })

        """for param in patent["fields"]["F_TRIZ_PARAMS"]:
            if param not in f_parameters_list:
                f_parameters_list[param] = []
            f_parameters_list[param].append(len(database)-1)
        for param in patent["fields"]["S_TRIZ_PARAMS"]:
            if param not in s_parameters_list:
                s_parameters_list[param] = []
            s_parameters_list[param].append(len(database)-1)"""

    print(len(response["hits"]["hits"]), len(database))

    if start+1000 < response["hits"]["total"]["value"]:
        download_week(early, late,start+1000)


database = []
f_parameters_list = {}
s_parameters_list = {}

week = timedelta(days=7)
date = datetime(2022, 10, 12)  # 2022-10-11
date = datetime(2017, 8, 2)  # 2022-10-11
date_minus_7 = date-week
first_publication = datetime(2002, 1, 1)  # 2002-01-01


counter_save=0
while date>=first_publication:
    print(date_minus_7.strftime("%Y-%m-%d"), "=>", date.strftime("%Y-%m-%d"))
    
    download_week(date_minus_7, date)
    date_minus_7-=week
    date-=week

    counter_save+=1
    if counter_save==100:
        counter_save=0
        #np.save("f_parameters_list", f_parameters_list)
        #np.save("s_parameters_list", s_parameters_list)

        f = open("all_database.json", "w")
        f.write(json.dumps(database, separators=(',', ':')))
        f.close()

        print("last save : ", date_minus_7.strftime("%Y-%m-%d"), "=>", date.strftime("%Y-%m-%d"))


"""print(database)
print(f_parameters_list)
print(s_parameters_list)"""

print(len(database))

#np.save("f_parameters_list", f_parameters_list)
#np.save("s_parameters_list", s_parameters_list)

f = open("all_database.json", "w")
f.write(json.dumps(database, separators=(',', ':')))
f.close()
