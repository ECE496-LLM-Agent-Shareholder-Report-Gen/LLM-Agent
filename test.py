import json

d = {}

file_name = "saved_session.json"
with open(file_name, "w") as f:
    json.dump(d, f)