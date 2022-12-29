import json

with open('D:\OTTO\Data\\tmpjs.jsonl', 'r') as json_file:
    data = json.load(json_file)

print(data['1'][0])