import json

with open("classes.json", "r") as f:
    class_indices = json.load(f)

print("Classes mapeadas:")
print(class_indices)
