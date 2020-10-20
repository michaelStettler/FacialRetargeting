import numpy as np
import json

with open("C:/Users/Michael/PycharmProjects/FacialRetargeting/configs/David_to_Louise_v2.json") as f:
    config = json.load(f)

list = np.array(config['vrts_pos']).astype(int)
print(list)
print(list[0])