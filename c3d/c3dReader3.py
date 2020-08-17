import ctypes
import copy
from c3d import vector3

"""
c3dReader for python 3
"""


class BinaryFileReader(object):
    """ BinaryFileReader is a class to help ease the reading of binary data from a file
        With easy method to read Bytes, Ints, Longs, Floats and Strings
    """
    def __init__(self, filePath=None):
        """ Constructor

            Args:
                filePath (str): the path of a the file to load in (OPTIONAL)
        """
        self._handle = None
        self._filePath = None
        if filePath is not None:
            self.openFile(filePath)

    def openFile(self, filePath):
        """ Open the give file for reading

            Args:
                filePath (str): the path of a the file to load in (OPTIONAL)
        """
        self._filePath = filePath
        self._handle = open(filePath, "rb")

    def closeFile(self):
        """ Close the current file
        """
        self._handle.close()

    def seek(self, seekPos):
        """ To go the position in the file

            Args:
                seekPos (int): the amount of bytes to go to in the file
        """
        self._handle.seek(seekPos)

    def tell(self):
        """ Get the current reader position in the file

            Returns:
                an Int of the amount of bytes the reader is currently in the file
        
        """
        return self._handle.tell()

    def readStringFromByte(self, length):
        """ Method read a set length string from a binary file

            Args:
                length (int)        : the length of the string to read in

            Returns:
                    A string from the c3d file, the length of which matches in
                    the input length
        """
        word = []
        paramName = ""
        for _ in range(length):
            word.append(self.readByte())
        paramName = paramName.join(map(chr, word))
        return paramName

    def readByte(self):
        """ Method read a single byte from a binary file

            Returns:
                    The byte value
        """
        tempByte = ctypes.c_byte()
        self._handle.readinto(tempByte)
        return tempByte.value

    def readInt(self):
        """ Method read a single int from a binary file

            Returns:
                    The int value
        """
        tmpInt = ctypes.c_int16()
        self._handle.readinto(tmpInt)
        return tmpInt.value

    def readLong(self):
        """ Method read a single long int from a binary file

            Returns:
                    The long int value
        """
        tmpLong = ctypes.c_long()
        self._handle.readinto(tmpLong)
        return tmpLong.value

    def readFloat(self):
        """ Method read a single float from a binary file
            Returns:
                    The float value
        """
        tmpFloat = ctypes.c_float()
        self._handle.readinto(tmpFloat)
        return tmpFloat.value


class C3DReader(object):
    """ Class to handle and deal with the reading in of C3D files, and allowing
        a easy API to access the marker data, and the parameters with the data
    """
    __c3dFileKeyValue__ = 80

    def __init__(self, filePath=None):
        """ Constructor, sets all the internal values to the default of 0 
            and if a file path is supplied, will load the c3d file

            Args:
                filePath (str): the path of a c3d file to load in (OPTIONAL)
        """
        self._parameterBlock = 0
        self._markerCount = 0
        self._analogMesasurements = 0
        self._firstFrame = 0
        self._lastFrame = 0
        self._maxFrameGap = 0
        self._scaleFactor = 0.0
        self._dataStart = 0
        self._sampleRate = 0
        self._frameRate = 0.0
        self._filePath = ""

        self._numberOfparameters = 0
        self._processorType = 0
        self._parameterDict = {}
        self._paramGroupToName = {}
        self.reset()

        if filePath is not None:
            self.loadFile(filePath)

    def reset(self):
        """ resets all the inernals values to their defaults
        """
        self._parameterBlock = 0
        self._markerCount = 0
        self._analogMesasurements = 0
        self._firstFrame = 0
        self._lastFrame = 0
        self._maxFrameGap = 0
        self._scaleFactor = 0.0
        self._dataStart = 0
        self._sampleRate = 0
        self._frameRate = 0.0
        self._filePath = ""

        self._numberOfparameters = 0
        self._processorType = 0
        self._parameterDict = {}
        self._paramGroupToName = {}

    def loadFile(self, filePath):
        """ Load the given c3d file into the object. Will read the headers and
            parameter sections of the file, but will not read any of the actual
            data, as that is called on a frame basis.

            Args:
                filePath (str): the path of a c3d file to load in
        """
        self._filePath = filePath
        fileObj = BinaryFileReader(filePath)
        self.__readHeader(fileObj)
        self.__readparameters(fileObj)
        fileObj.closeFile()

    def __readHeader(self, handle):
        """ Internal Method to read the header of a c3d file and populate the
            object with the data

            Args:
                handle (binaryFile): file handle object to the open c3d file
            Raises:
                ValueError: If the file is not a valid c3d File
        """
        handle.seek(0)
        self.__parameterBlock = handle.readByte()  # word 1
        if handle.readByte() != self.__c3dFileKeyValue__:
            raise ValueError("Not a valid C3D file")
        self._markerCount = handle.readInt()  # word 2
        self._analogMesasurements = handle.readInt()  # word 3
        self._firstFrame = handle.readInt()  # word 4
        self._lastFrame = handle.readInt()  # word 5
        self._maxFrameGap = handle.readInt()  # word 6
        self._scaleFactor = handle.readFloat()  # word 7-8
        self._dataStart = handle.readInt()  # word 9
        self._sampleRate = handle.readInt()  # word 10
        self._frameRate = handle.readFloat()  # word 11-12
        # More can be added as needed

    def __newParam(self, handle, name, groupId):
        """ Internal Method to create a new parameter set from the parameter
            section of the c3d file. Will create the internal data structor
            for the new parameter

            Args:
                handle (binaryFile)    : file handle object to the open c3d file
                name (str)        : name of the new parameter
                groupId (int)    : number index of the parameter group
        """
        self._parameterDict[name] = {}
        self._paramGroupToName[groupId] = name
        self.__readDescription(handle)

    def __readDescription(self, handle):
        """ Internal method to read the a parameter description
            Current it does nothing with this data, but it can be stored.

            Args:
                handle (binaryFile) : the c3d file to read
        """
        paramDescirptionLength = handle.readByte()
        if paramDescirptionLength == 0:
            return
        handle.readStringFromByte(paramDescirptionLength)

    def __readparameters(self, handle):
        """ Internal method to read the entire parameter block of a c3d file

            Args:
                handle (binaryFile) : the c3d file to read
        """
        handle.seek(512)
        handle.readByte()
        handle.readByte()
        self._numberOfparameters = handle.readByte()
        self._processorType = handle.readByte()
        self._parameterDict = {}
        self._paramGroupToName = {}

        while 1:
            if self.__readParamBlock(handle) is False:
                break

    def __readParamBlock(self, handle):
        """ Internal method to read the a single parameter of a c3d file

            Args:
                handle (binaryFile) : the c3d file to read
        """
        value = None
        nameLength = abs(handle.readByte())
        groupId = abs(handle.readByte())
        groupName = handle.readStringFromByte(nameLength)
        currentFilePos = handle.tell()
        bytesToNextGroup = handle.readInt()

        if bytesToNextGroup == 0:
            return False

        if groupId not in self._paramGroupToName.keys():
            self.__newParam(handle, groupName, groupId)
            return
#         import pdb; pdb.set_trace()
        dataType = handle.readByte()
        dataDimensions = handle.readByte()

        # Its a string
        if dataType == -1:
            stringSize = handle.readByte()
            # Its a String
            if dataDimensions > 1:
                value = []
                arraySize = handle.readByte()
                for _ in range(arraySize):
                    value.append(handle.readStringFromByte(stringSize).strip())
            else:
                value = handle.readStringFromByte(stringSize).strip()
        # Its a scaler
        elif dataDimensions == 0:
            value = self.__readPropertyValue(handle, dataType)
        else:
            value = []
            for _ in range(dataDimensions):
                value.append(self.__readPropertyValue(handle, dataType))

        self._parameterDict[self._paramGroupToName[groupId]][groupName] = value
        handle.seek(currentFilePos + bytesToNextGroup)

    def __readPropertyValue(self, handle, propertyType):
        """ Internal Method to read type of value from the c3d file.

            Support reading: bytes, int's and floats. The type is passed in
            by the propertyType arg.

            Args:
                handle (binaryFile)   : file handle object to the open c3d file
                propertyType (int)    : the type of value to read in, values are:
                                       -1 - String
                                        1 - Byte
                                        2 - Int
                                        4 - Flaot

            Returns:
                    The read file data in the requested value

            Raises:
                ValueError: the property type is of an unknown value
        """
        if propertyType == -1:
            nameLength = handle.readByte()
            propertyValue = handle.readStringFromByte(nameLength)
        elif propertyType == 1:
            propertyValue = handle.readByte()
        elif propertyType == 2:
            propertyValue = handle.readInt()
        elif propertyType == 4:
            propertyValue = handle.readFloat()
        else:
            raise ValueError("Unkown value given: %r" % propertyType)
        return propertyValue

    def __readMarkerData(self, handle):
        """ Internal Method to the 3 float values that make up a marker location,
            and return a vector3 with the values.

            TODO: Add in order transform

            Args:
                handle (binaryFile)        : file handle object to the open c3d file

            Returns:
                    A vector 3 object with the X, Y and Z coordinates of the marker
        """
        tempX = handle.readFloat()
        tempY = handle.readFloat()
        tempZ = handle.readFloat()
        handle.readFloat()
        newVect = vector3.Vector3(tempX, tempY, tempZ)
        return newVect

    def readFrame(self, frameNumber):
        """ Read a single frame of data for each marker.

            Args:
                frameNumber (int)    : The frame to get the data from.

            Returns:
                    A list of vector3 objects, one for each of the markers.

            Raises:
                ValueError: if the given frame is not within the frame range of
                            the c3d file
        """
        if frameNumber < self._firstFrame or frameNumber > self._lastFrame:
            raise ValueError("Invalid Frame Number")
        startOfDataBlock = (self._parameterDict["POINT"]["DATA_START"] - 1) * 512

        inspectFrame = frameNumber - self._firstFrame
        byteFrame = (self._markerCount * 16) * inspectFrame

        fileObj = BinaryFileReader(self._filePath)
        fileObj.seek(startOfDataBlock + byteFrame)
        markerData = []
        for _ in range(self._markerCount):
            markerData.append(self.__readMarkerData(fileObj))
        # fileObj.close()
        return markerData

    def iterFrame(self, start=None, end=None, iterJump=1):
        """ Iterate through all the frames in the c3d file. Allows for
            specifying    the start and stop frame as well as frame skip
            number

            Args:
                start (int)        : The start frame to use (Optional)
                end (int)        : The end frame to use (Optional)
                iterJump (int)    : The number of frames to skip (Optional)

            Returns:
                    a list of lists of vector3 objects, one for each of the 
                    markers at each frame
        """
        firstFrame = start or self._firstFrame
        lastFrame = end or (self._lastFrame + 1)
        for frame in range(firstFrame, lastFrame, iterJump):
            yield self.readFrame(frame)

    def getMarkerCount(self):
        """ Get the marker count

            Returns:
                    The number of markers in the c3d file as a int
        """
        return self._markerCount

    def getMarkers(self):
        """ Get a list of the markers names from the c3d file

            Returns:
                    A list of marker names from the loaded c3d file.
        """
        if self._markerCount == 0:
            return []
        return list(self._parameterDict["POINT"]["LABELS"])

    def getFrameStart(self):
        """ Get the frame start of the c3d file

            Returns:
                    The frame start as a int
        """
        return self._firstFrame

    def getFrameEnd(self):
        """ Get the frame end of the c3d file

            Returns:
                    The end start as a int
        """
        return self._lastFrame

    def getSampleRate(self):
        """ Get the sample ratge of the c3d file

            Returns:
                    The c3d file sameple rate as a float
        """
        return self._frameRate

    def getLoadedFilePath(self):
        """ Get the path of the file that the reader is reading from

            Returns:
                    String file Path
        """
        return self._filePath

    def listparameters(self):
        """ List all the parameters in the c3d

            Returns:
                    a string list of all the parameters in the c3d file
        """
        return list(self._parameterDict.keys())

    def listparametersProperties(self, parameter):
        """ List all the properties in the given parameter.

            Args:
                parameter (str)    : The parameter that you want the properties of

            Returns:
                    A list of properties within the parameters as strings
        """
        if parameter not in self._parameterDict.keys():
            raise ValueError("Parameter not found in file")
        return list(self._parameterDict[parameter].keys())

    def getparameterDict(self):
        """ Returns a copy of the internal parameter dict

            Returns:
                    A two level nested dict of parameter, property & value key pairs
        """
        return copy.deepcopy(self._parameterDict)

    def getparameter(self, parameter, pramProperty):
        """ Get the value of the property within the parameter.
            Can return a string, int, long or float depending on what the value
            is

            Args:
                parameter (str)    : The parameter that you want the properties of
                pramProperty (str)    : The property from the parameter

            Returns:
                    A list of properties within the parameter as strings
        """
        if parameter not in self._parameterDict.keys():
            raise AttributeError("parameter %r not found in file" % parameter)
        if pramProperty not in self._parameterDict[parameter].keys:
            raise AttributeError("Property %r not found in %s" % (pramProperty, parameter))
        return self._parameterDict[parameter][pramProperty]
