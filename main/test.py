import json

data={'a':1,'b':2}
data=json.dumps(data)
with open('/home/qiaodawang19/otto/data/tmp.jsonl', 'w') as json_file:
    json_file.write(data)
