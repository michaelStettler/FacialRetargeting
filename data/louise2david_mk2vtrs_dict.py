# create dictionary of vertices index to match markers (they have to match semantically)
"""
and thanks to python 2.7 in maya this does not keep it's order...
"""
mk2vtx = {
    'leftBrow1': 2912,  # 0
    'leftBrow2': 2589,
    'leftBrow3': 2909,
    'leftBrow4': 2927,
    'RightBrow1': 399,
    'RightBrow2': 76,  # 5
    'RightBrow3': 396,
    'RightBrow4': 414,
    'Nose1': 1779,
    'Nose2': 155,
    'Nose3': 825,  # 10
    'Nose4': 2195,
    'Nose5': 3338,
    'Nose6': 2668,
    'Nose7': 23,
    'Nose8': 34,  # 15
    'UpperMouth1': 1993,
    'UpperMouth2': 2138,
    'UpperMouth3': 2202,
    'UpperMouth4': 4634,
    'UpperMouth5': 4489,  # 20
    'LowerMouth1': 2054,
    'LowerMouth2': 2209,
    'LowerMouth3': 4550,
    'LowerMouth4': 1805,
    'LeftOrbi1': 2770,  # 25
    'RightOrbi1': 257,
    'RightOrbi2': 831,
    'RightCheek1': 406,
    'RightCheek3': 84,
    'LeftCheek1': 2919,  # 30
    'LeftCheek2': 3572,
    'LeftCheek3': 2597,
    'LeftJaw1': 2845,
    'LeftJaw2': 2772,
    'RightJaw1': 332,  # 35
    'RightJaw2': 259,
    'LeftEye1': 3393,
    'Head1': None,  # 38
    'Head2': None,  # 39
    'Head3': None,  # 40
    'LeftOrbi2': 3344,
    'RightEye1': 880,
    'RightCheek2': 539,
    'Head4': None,  # 44
}