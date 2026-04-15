import torch

path = "skin_detector_v2.pth"

data = torch.load(path, map_location="cpu")
print("Тип:", type(data))

if isinstance(data, dict):
    print("\nКлючи:", data.keys())
    if "state_dict" in data:
        print("\nВесов в state_dict:", len(data["state_dict"]))
        print("Первые 20 ключей:")
        for k in list(data["state_dict"].keys())[:20]:
            print(k)
    else:
        print("\nПервые 20 ключей:")
        for k in list(data.keys())[:20]:
            print(k)
else:
    print(data)
