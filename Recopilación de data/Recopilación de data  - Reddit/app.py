import pathlib
import csv
import requests
import praw
import pandas as pd
client_id = "UsPvWxuIRoXcug"
client_secret = "VK5TJqKTaL9rYWZqTsRGMYUvwWZX-g"
user_agent = "Testing_api"

from datetime import datetime
from django.utils import timezone
from dateutil import tz
import pytz
local_tz = pytz.timezone('America/Lima')

r = praw.Reddit(
    user_agent= user_agent,
    client_id=client_id,
    client_secret=client_secret
    )
    
PATH = pathlib.Path(__file__)
DATA_PATH = PATH.joinpath("../csv posts").resolve()

CSV_PATH = DATA_PATH.joinpath('krvwkf.csv')

lst_id = list()
file = open(CSV_PATH, "rU")
reader = csv.reader(file, delimiter=',')
for row in reader:
    for column in row:
        lst_id.append(column)


def get_reddit_data(ids):
    params = {}
    ids[0] = 't1_' + ids[0]
    submission = r.info(',t1_'.join(ids).split())
    return submission

lst_ds = []    

while lst_id:
    data = get_reddit_data(lst_id[:100])
    print(len(lst_id))
    for i, x in enumerate(data):
        dt = datetime.fromtimestamp(x.created_utc).astimezone(tz=local_tz)
        if not x.body == '[eliminado]':
            cm = {
            "created": dt.strftime('%Y-%m-%d %H:%M:%S'),
            "created_hour" : dt.strftime('%Y-%m-%d %H'),
            "comment" : " ".join(x.body.replace(u"\u2018", "'").replace(u"\u2019", "'").replace(u'\u201c', '"').replace(u"\u201d","").split()) ,
            "post" : 'krvwkf',
            "score" : x.score,
            "id" : x.id
            } 

            lst_ds.append(cm)
    del lst_id[:100]  # (Mif rake sure the call was successful before removing the batch like this)

df = pd.DataFrame(lst_ds)
CSV_RESULT_PATH = DATA_PATH.joinpath('krvwkf_result.csv')
df.to_csv(CSV_RESULT_PATH, index = False, header=True)
