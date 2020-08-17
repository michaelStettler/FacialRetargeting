import maya.cmds as cmds
import numpy as np
import c3d

reader = c3d.Reader(open('C:\Users\David\Documents\Viccon_david\FacialExpressionsTracking\Test\NewSession\David_Trail36.c3d', 'rb'))
for i, points, analog in reader.read_frames(0,10):
    print('frame {}: point {}, analog {}'.format(
        i, points.shape, analog.shape))
print()