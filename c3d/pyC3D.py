import math
import pickle
import numpy as np
from norm_pos import getHeadPos, RigidTransform3D

#from maya import utils as mayaUtils
from pyc3d import c3dReader, vector3

"""
David:  - Debugged C3D Loader
        - added normalization: when calling C3DReader set normalize=True to normalize
        - normalize using Rigid Transformation with SVD

"""

class C3DReader(object):
    def __init__(self, filePath=None, launchUI=True,normalize=False):
        super(C3DReader, self).__init__()
        self.allMarkers = []
        if filePath is not None:
            self.loadFile(filePath, normalize)
            return
        if launchUI:
            # TODO: Add UI
            pass

    def loadFile(
                self,
                filePath,normalize,
                startFrame=None,
                endFrame=None,
                zeroStart=None
                ):
        self.fileData = c3dReader.C3DReader(filePath)

        """set scene frame rate to 120 and update in maya"""
        sceneFrameRate = 120
        frameRate = "{0}fps".format(sceneFrameRate)
        cmds.currentUnit(time=frameRate)


        self.frameRate = self.fileData.getSampleRate()
        self.skipFrame = (self.frameRate / sceneFrameRate)
        self.startFrame = (sceneFrameRate / self.frameRate) * self.fileData.getFrameStart()
        self.endFrame = (sceneFrameRate / self.frameRate) * self.fileData.getFrameEnd()
        cmds.playbackOptions(minTime=self.startFrame, maxTime=self.endFrame)

        tempMakers = self.fileData.getMarkers()
        self.allMarkers = []
        cmds.spaceLocator(p=[0, 0, 0], name="C3DOpticalRoot")
        for _marker in tempMakers:
            self.allMarkers.append(_marker.replace(":", "_"))
        self.__createMarkers()
        self.__mapFrameData(normalize=normalize)

    def createCube(self, scale=0.2):
        temp = cmds.curve(d=1,
                            p=[
                                (1, 1, 1), (1, 1, -1), (-1, 1, -1), (-1, 1, 1) , (1, 1, 1),
                                (1, -1, 1), (1, -1, -1), (-1, -1, -1), (-1, -1, 1) , (1, -1, 1),
                                (1, -1, -1), (1, -1, 1), (1, 1, 1) , (1, 1, -1), (1, -1, -1),
                                (-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1), (-1, -1, -1)
                            ]
                        )
        cmds.setAttr("%s.scaleX" % temp, scale)
        cmds.setAttr("%s.scaleY" % temp, scale)
        cmds.setAttr("%s.scaleZ" % temp, scale)
        return temp

    def __createMarkers(self):
        for _marker in self.allMarkers:
            # "Making %r" % _marker
            Temp = self.createCube()
            cmds.rename(Temp, _marker)
            cmds.parent(_marker, "C3DOpticalRoot")

    def __mapFrameData(self, zeroStart=None, normalize=False):
        currentFrame = zeroStart or self.startFrame
        frameJump = int(math.ceil(self.skipFrame))
        if normalize:
            # normalized position stored in pickle file
            normPos = pickle.load(open(r'E:\maya\david_mocap\data\Normalized_head_position', 'rb'))
        for data in self.fileData.iterFrame(iterJump=frameJump):
            if normalize:
                # get head positions necessary for normalization and perform SVD to get rotation and translation
                head1 = head2 = head3 = head4 = None
                if 'Head1' in self.allMarkers:
                    head1=data[self.allMarkers.index('Head1')]
                if 'Head2' in self.allMarkers:
                    head2=data[self.allMarkers.index('Head2')]
                if 'Head3' in self.allMarkers:
                    head3=data[self.allMarkers.index('Head3')]
                if 'Head4' in self.allMarkers:
                    head4=data[self.allMarkers.index('Head4')]
                headPos = getHeadPos(head1, head2, head3, head4)
                R, t = RigidTransform3D(np.array(headPos), np.array(normPos))

            for idx in xrange(len(self.allMarkers)):
                if normalize:
                    # normalize all markers with respect to head markers (if at least 3 existing)
                    new = np.dot(np.array([data[idx].x, data[idx].y, data[idx].z]), R.T) + t
                    data[idx].setX(new[0])
                    data[idx].setY(new[1])
                    data[idx].setZ(new[2])
                cmds.setKeyframe(self.allMarkers[idx], v=data[idx].x / 100, time=currentFrame, at='translateX')
                cmds.setKeyframe(self.allMarkers[idx], v=-data[idx].y / 100, time=currentFrame, at='translateZ')
                cmds.setKeyframe(self.allMarkers[idx], v=data[idx].z / 100, time=currentFrame, at='translateY')
            currentFrame += 1