from pyc3d import pyC3D
from maya import cmds
import os

def saveFile(file):
    cmds.file(rename=file)
    cmds.file(save=True, type="mayaBinary")
    cmds.file(newFile=True)

for filename in os.listdir( 'C:\Users\David\Documents\Viccon_david\FacialExpressionsTracking\Test\NewSession_labeled'):
    if filename.endswith(".c3d"):
        print(filename)
        c3dFile = pyC3D.C3DReader(
            'C:\Users\David\Documents\Viccon_david\FacialExpressionsTracking\Test\NewSession_labeled\\' + filename,
            normalize=True)
        saveFile(os.path.splitext(filename)[0])