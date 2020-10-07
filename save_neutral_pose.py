import numpy as np
import os
from utils.load_data import load_c3d_file

# declare variables
path = 'D:/MoCap_Data/David/NewSession_labeled/'
file = 'NeutralTrail14.c3d'
save_folder = 'data/'
save_name = 'David_neutral_pose'
neutral_frame = 900
template_labels = ['LeftBrow1', 'LeftBrow2', 'LeftBrow3', 'LeftBrow4', 'RightBrow1', 'RightBrow2', 'RightBrow3',
                   'RightBrow4', 'Nose1', 'Nose2', 'Nose3', 'Nose4', 'Nose5', 'Nose6', 'Nose7', 'Nose8',
                   'UpperMouth1', 'UpperMouth2', 'UpperMouth3', 'UpperMouth4', 'UpperMouth5', 'LowerMouth1',
                   'LowerMouth2', 'LowerMouth3', 'LowerMouth4', 'LeftOrbi1', 'LeftOrbi2', 'RightOrbi1', 'RightOrbi2',
                   'LeftCheek1', 'LeftCheek2', 'LeftCheek3', 'RightCheek1', 'RightCheek2', 'RightCheek3',
                   'LeftJaw1', 'LeftJaw2', 'RightJaw1', 'RightJaw2', 'LeftEye1', 'RightEye1', 'Head1', 'Head2',
                   'Head3', 'Head4']

# load sequence
data, labels = load_c3d_file(os.path.join(path, file),
                             template_labels=template_labels,
                             get_labels=True,
                             verbose=True)
print("labels", len(labels))
print(labels)
print("shape data[neutral_frame]", np.shape(data[neutral_frame]))
print(data[neutral_frame])

# save
np.save(os.path.join(save_folder, save_name), data[neutral_frame])