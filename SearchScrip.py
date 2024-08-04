import urllib.request
import json
import pandas as pd
import os

url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"


def generateScripMaster():
    with urllib.request.urlopen(url) as response:
        data = json.load(response)
        df=pd.DataFrame.from_dict(data, orient='columns')
        # df=df.loc[df['exch_seg'] == "NSE"]
        df = df.loc[df['symbol'].str.endswith("-EQ")].sort_values(by=['symbol'])
        df.to_csv("ScripMaster.csv",index=False)

def getNSEScripToken(symbol):
    if os.path.exists("ScripMaster.csv"):
        df = pd.read_csv("ScripMaster.csv")
        df = df.loc[df['name'] == symbol]
        token=df['token'].iloc[0]
        # print(token)
        return int(token)
    else:
        generateScripMaster()
        getNSEScripToken(symbol)
    return 0

getNSEScripToken("CDSL")