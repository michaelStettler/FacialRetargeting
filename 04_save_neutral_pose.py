import numpy as np
import os
from utils.load_data import load_c3d_file
import json

# load and define parameters
with open("C:/Users/Michael/PycharmProjects/FacialRetargeting/configs/David_to_Louise_v2.json") as f:
    config = json.load(f)

# load sequence
data, labels = load_c3d_file(os.path.join(config['mocap_folder'], config['neutral_sequence']),
                             template_labels=config['template_labels'],
                             get_labels=True,
                             verbose=True)
print("labels", len(labels))
print(labels)
print("shape data[neutral_frame]", np.shape(data[int(config['neutral_frame'])]))
print(data[int(config['neutral_frame'])])

# save
np.save(os.path.join(config['python_data_path'], config['neutral_pose_positions']), data[int(config['neutral_frame'])])