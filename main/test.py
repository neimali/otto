import json

with open('/home/qiaodawang19/otto/data/sentences.jsonl', 'r') as json_file:
    data = json.load(json_file)
print(data)
