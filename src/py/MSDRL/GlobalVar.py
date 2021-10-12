import numpy as np

LABELS = {
    "u" : ['PNS','ANS','A','UR6apex','UR3apex','U1apex','UL3apex','UL6apex','UR6d','UR6m','UR3tip','UItip','UL3tip','UL6m','UL6d'],
    "l" : ['RCo','RGo','LR6apex','LR7apex','L1apex','Me','Gn','Pog','B','LL6apex','LL7apex','LGo','LCo','LR6d','LR6m','LItip','LL6m','LL6d'],
    "cb" :['Ba','S','N']
}

MOVEMENT_MATRIX = np.array([
    [1,0,0],  # MoveUp
    [-1,0,0], # MoveDown
    [0,1,0],  # MoveBack
    [0,-1,0], # MoveFront
    [0,0,1],  # MoveLeft
    [0,0,-1], # MoveRight
])

MOVEMENT_ID = ["Up", "Down", "Back", "Front", "Left", "Right"]


def GetTargetOutputFromAction(action):
    target = np.zeros((1,len(MOVEMENT_MATRIX)))[0]
    target[action] = 1
    return target