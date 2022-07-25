import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS_TO_TRAIN = []
GROUP_LABELS = {
    'CB' : ['Ba', 'S', 'N', 'RPo', 'LPo', 'RFZyg', 'LFZyg', 'C2', 'C3', 'C4'],

    'U' : ['RInfOr', 'LInfOr', 'LMZyg', 'RPF', 'LPF', 'PNS', 'ANS', 'A', 'UR3O', 'UR1O', 'UL3O', 'UR6DB', 'UR6MB', 'UL6MB', 'UL6DB', 'IF', 'ROr', 'LOr', 'RMZyg', 'RNC', 'LNC', 'UR7O', 'UR5O', 'UR4O', 'UR2O', 'UL1O', 'UL2O', 'UL4O', 'UL5O', 'UL7O', 'UL7R', 'UL5R', 'UL4R', 'UL2R', 'UL1R', 'UR2R', 'UR4R', 'UR5R', 'UR7R', 'UR6MP', 'UL6MP', 'UL6R', 'UR6R', 'UR6O', 'UL6O', 'UL3R', 'UR3R', 'UR1R'],

    'L' : ['RCo', 'RGo', 'Me', 'Gn', 'Pog', 'PogL', 'B', 'LGo', 'LCo', 'LR1O', 'LL6MB', 'LL6DB', 'LR6MB', 'LR6DB', 'LAF', 'LAE', 'RAF', 'RAE', 'LMCo', 'LLCo', 'RMCo', 'RLCo', 'RMeF', 'LMeF', 'RSig', 'RPRa', 'RARa', 'LSig', 'LARa', 'LPRa', 'LR7R', 'LR5R', 'LR4R', 'LR3R', 'LL3R', 'LL4R', 'LL5R', 'LL7R', 'LL7O', 'LL5O', 'LL4O', 'LL3O', 'LL2O', 'LL1O', 'LR2O', 'LR3O', 'LR4O', 'LR5O', 'LR7O', 'LL6R', 'LR6R', 'LL6O', 'LR6O', 'LR1R', 'LL1R', 'LL2R', 'LR2R'],

    'CI' : ['UR3OIP','UL3OIP','UR3RIP','UL3RIP'],

    'TMJ' : ['AF', 'AE']
}

# print(len(GROUP_LABELS['CB']) + len(GROUP_LABELS['U']) + len(GROUP_LABELS['L']) + len(GROUP_LABELS['CI']))

# LABELS_TO_TRAIN = ['Ba', 'S', 'N', 'RPo', 'LPo', 'RFZyg', 'LFZyg']

# LABELS_TO_TRAIN = ['ROr','LOr','LMZyg','RMZyg','RPF','LPF','RNC','LNC','UR6O','UL6O','UR6R','UL6R','UR4O','UR4O','UL4O','UL4R','UR3O','UL3O','UR3R','UL3R','UR1O','UL1O','UR1R','UL1R','ANS','PNS','A']
# LABELS_TO_TRAIN = ['RCo', 'LCo', 'RGo', 'LGo', 'Me', 'Gn', 'Pog', 'B', 'RPRa', 'LPRa', 'RARa', 'LARa', 'LR6O', 'LL6O', 'LR6R', 'LL6R', 'LR3O', 'LL3O', 'LR3R', 'LL3R', 'LR1O', 'LL1O', 'LR1R', 'LL1R']



# new_LABELS = []
# for label in GROUP_LABELS['L']:
#     if label not in LABELS_TO_TRAIN:
#         new_LABELS.append(label)

# LABELS_TO_TRAIN = new_LABELS


LABELS_TO_TRAIN = ['AF','AE']

LABEL_GROUPES = {}
LABELS = []
for group,labels in GROUP_LABELS.items():
    for label in labels:
        LABEL_GROUPES[label] = group
        LABELS.append(label)
        # LABELS_TO_TRAIN.append(label)

# print(len(LABELS))

SCALE_KEYS = ['1','0-3']


MOVEMENT_MATRIX_6 = np.array([
    [1,0,0],  # MoveUp
    [-1,0,0], # MoveDown
    [0,1,0],  # MoveBack
    [0,-1,0], # MoveFront
    [0,0,1],  # MoveLeft
    [0,0,-1], # MoveRight
])
MOVEMENT_ID_6 = [
    "Up",
    "Down",
    "Back",
    "Front",
    "Left",
    "Right"
]

MOVEMENTS = {
    "id" : MOVEMENT_ID_6,
    "mat" : MOVEMENT_MATRIX_6

}


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def GetTargetOutputFromAction(mov_mat,action):
    target = np.zeros((1,len(mov_mat)))[0]
    target[action] = 1
    return target