from typing import Sequence, Tuple, Union
import sys

import SimpleITK as sitk
import numpy as np
from scipy.sparse.construct import random
from sklearn import neighbors

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn


# ----- MONAI ------
# from monai.losses import DiceCELoss
# from monai.inferers import sliding_window_inference
from monai.transforms import (
    transform,
    Compose,
    AddChannel,
    ScaleIntensity,
    SpatialCrop,
    BorderPad,
)
# from torchvision import models

from utils import*

# #####################################
#  Networks
# #####################################

class DRLnet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        dropout_rate: float = 0.0,
    ) -> None:
        super(DRLnet, self).__init__()
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        self.conv1 = nn.Conv3d(
            in_channels, 
            out_channels = 32, 
            kernel_size = 4, 
        )
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(
            in_channels = 32, 
            out_channels = 42, 
            kernel_size = 3, 
        )
        self.pool2 = nn.MaxPool3d(2)

        self.fc0 = nn.Linear(115248,512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 6)

        # self.pool1 = nn.MaxPool3d(kernel_size = 2)

    def forward(self,x):
        # print(x.size())
        x=self.pool1(F.relu(self.conv1(x)))
        # print(x.size())
        x=self.pool2(F.relu(self.conv2(x)))
        # print(x.size())
        x = torch.flatten(x, 1)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.log_softmax(self.fc3(x), dim=1)
        return output



# #####################################
#  Environement
# #####################################

class Environement :
    def __init__(
        self,
        images_path,
        padding,
    ) -> None:
        """
        Args:
            images_path : path of the image with all the different scale,
            landmark_fiducial : path of the fiducial list linked with the image,
        """
        self.padding = padding.astype(np.int16)
        print(self.padding)
        self.LoadImages(images_path)
        self.ResetLandmarks()


    def LoadImages(self,images_path):
        transform = Compose([AddChannel(),BorderPad(spatial_border=self.padding.tolist()),ScaleIntensity(minv = 0.0, maxv = 1.0, factor = None)])

        data = []
        sizes = []
        spacings = []
        origins = []
        
        for path in images_path:
            img = sitk.ReadImage(path)
            # sizes.append(np.array(img.GetSize()))
            spacings.append(np.array(img.GetSpacing()))
            origin = img.GetOrigin()
            origins.append(np.array([origin[2],origin[1],origin[0]]))
            img_ar = sitk.GetArrayFromImage(img)
            sizes.append(np.shape(img_ar))
            data.append(torch.from_numpy(transform(img_ar)))

        self.dim = len(data)
        self.data = data
        self.sizes = sizes
        self.spacings = spacings
        self.origins = origins

    def ResetLandmarks(self):
        dim_lm = []
        for i in range(self.dim):
            dim_lm.append({})
        self.dim_landmarks = dim_lm


    def LoadLandmarks(self,fiducial_path):
        fcsv_lm_lst = ReadFCSV(fiducial_path)
        for lm in fcsv_lm_lst :
            lm_ph_coord = np.array([float(lm["z"]),float(lm["y"]),float(lm["x"])])
            for i in range(self.dim):
                lm_coord = (lm_ph_coord+ abs(self.origins[i]))/self.spacings[i]
                lm_coord = lm_coord.astype(int)
                self.dim_landmarks[i][lm["label"]] = lm_coord

    def GetSize(self,dim):
        return self.sizes[dim]

    def GetSpacing(self,dim):
        return self.spacings[dim]

    def GetLandmarkPos(self,dim,landmark):
        return self.dim_landmarks[dim][landmark]

    def GetL2DistFromLandmark(self, dim, position, target):
        label_pos = self.GetLandmarkPos(dim,target)
        return np.linalg.norm(position-label_pos)**2

    def GetZone(self,dim,center,crop_size):
        cropTransform = SpatialCrop(center.tolist(),crop_size)
        crop = cropTransform(self.data[dim])
        return crop


# #####################################
#  Agents
# #####################################


class TrainingAgent :
    def __init__(
        self,
        targeted_landmark,
        models,
        FOV = [32,32,32],
        gamma = 0.9,
        epsilon = 0.01,
        lr = 0.0005,
        batch_size = 10,
        max_mem_size = 100000,
        exp_end = 0.05,
        exp_dec = 5e-4,
        nbr_of_action = 6,
        start_pos_radius = 20,
        speed = 1,
        verbose = False
    ) -> None:
        """
        Args:
            environement : Environement in wich the target will progress,
            targeted_landmark : name of the landmark to target,
            models : List of network to train on each scale,
            FOV = [32,32,32] : region in the scan seen by the agent,
            gamma: Discount factor.
            epsilon: .
            lr: learning rate.
            batch_size:  .
            max_mem_size: size of the Agent memory.
            exp_end: minimum exploration porcentage .
            exp_dec: exploration porcentage reduction factor.
        """
        
        self.target = targeted_landmark
        self.scale_state = 0
        self.FOV = np.array(FOV, dtype=np.int16)
        self.verbose = verbose
        self.start_pos_radius = start_pos_radius
        self.position = np.array([0,0,0], dtype=np.int16)

        self.movement_matrix = np.array([
            [speed,0,0],  # MoveUp
            [-speed,0,0], # MoveDown
            [0,speed,0],  # MoveBack
            [0,-speed,0], # MoveFront
            [0,0,speed],  # MoveLeft
            [0,0,-speed], # MoveRight
        ])

        self.movement_id = ["Up", "Down", "Back", "Front", "Left", "Right"]


        #Brain
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.max_mem_size = max_mem_size
        self.exp_eps = 1.0
        self.exp_end = exp_end
        self.exp_dec = exp_dec
        self.loss_fn = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.models = models
        self.active_network = models[self.scale_state]

        optimizers = []
        for model in models:
            model.to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            optimizers.append(optimizer)
        self.optimizers = optimizers


        # Memory
        # self.mem_size = max_mem_size
        # self.mem_ctr = 0
        # self.state_mem = np.zeros((self.mem_size, *self.FOV), dtype=np.float32)
        # self.action_mem = np.zeros(self.mem_size, dtype=np.int32)
        # self.reward_mem = np.zeros(self.mem_size, dtype=np.float32)

    def SetEnvironement(self, environement : Environement): self.environement = environement

    def GoToScale(self,scale=0):
        self.position = (self.position*(self.environement.GetSpacing(self.scale_state)/self.environement.GetSpacing(scale))).astype(np.int16)
        self.scale_state = scale
        self.active_network = self.models[self.scale_state]

    def SetRandomPos(self):
        if self.scale_state == 0:
            rand_coord = np.random.randint(1, self.environement.GetSize(self.scale_state), dtype=np.int16)
        else:
            rand_coord = np.random.randint([1,1,1], self.start_pos_radius*2) - self.start_pos_radius
            rand_coord = self.environement.GetLandmarkPos(self.scale_state,self.target) + rand_coord
            rand_coord = np.where(rand_coord<0, 0, rand_coord)
            rand_coord = rand_coord.astype(np.int16)

        self.position = rand_coord


    def GetState(self):
        state = self.environement.GetZone(self.scale_state,self.position,self.FOV)
        return state

    def UpScale(self):
        if self.scale_state < self.environement.dim -1 :
            self.GoToScale(self.scale_state + 1)
        else:
            OUT_WARNING()

    def Move(self, movement_idx):
        new_pos = self.position + self.movement_matrix[movement_idx]
        if new_pos.all() > 0 and (new_pos < self.environement.GetSize(self.scale_state)).all():
            self.position += self.movement_matrix[movement_idx]
            if self.verbose:
                print("Moving ", self.movement_id[movement_idx])
        else:
            OUT_WARNING()

    def GetRewardLst(self):
        reward_lst = []
        agent_dist = self.environement.GetL2DistFromLandmark(self.scale_state,self.position,self.target)
        for move in self.movement_matrix:
            neighbor_coord = self.position + move
            dist_from_lm = self.environement.GetL2DistFromLandmark(self.scale_state,neighbor_coord,self.target)
            reward_lst.append(agent_dist - dist_from_lm)
        return reward_lst

    def GetBestMove(self):
        best_action = np.argmax(self.GetRewardLst())
        if self.verbose:
            print("Best move is ", self.movement_id[best_action])
        return best_action

    def e_greedy(self,wanted_action):
        rand = torch.rand(1)[0]
        if rand > self.exp_eps:
            action = wanted_action
        else:
            action = torch.randint(len(self.movement_id),(1,))[0]

        if self.exp_eps > self.exp_end: self.exp_eps-=self.exp_dec
        return action

    def Train(self,max_steps):
        print("Training :")
        # for dim in range(3):
        #     self.GoToScale(dim)
        self.SetRandomPos()
        steps_loss = 0.0
        reward_lst = []
        model = self.models[self.scale_state]
        model.train()
        for steps in range(max_steps):

            old_dist = self.environement.GetL2DistFromLandmark(self.scale_state,self.position,self.target)
            X = self.GetState().unsqueeze(0)
            X.to(self.device)
            x = model(X)
            # print(x)
            action = x.argmax(dim=1) 
            x = x[0][action]
            
            self.Move(self.e_greedy(action))
            new_dist = self.environement.GetL2DistFromLandmark(self.scale_state,self.position,self.target)
            reward = old_dist - new_dist

            # print("Reward :",reward)
            reward_lst.append(reward)

            Y = self.GetState().unsqueeze(0)
            Y.to(self.device)
            y = model(Y)
            action = y.argmax(dim=1) 
            y = reward + self.gamma * y[0][action]
            loss = self.loss_fn(y,x)
            steps_loss += loss.item()

            # print("Loss :", loss.item())

            # loss.backward()
            # self.optimizers[self.scale_state].step()

        val_loss = steps_loss / max_steps
        print("Loss :", val_loss)
        print("Rewards :", reward_lst)
        print("")


    def Validate(self,max_steps):
        print("Validation :")
        self.SetRandomPos()

        model = self.models[self.scale_state]
        model.eval()
        steps_loss = 0.0
        reward_lst = []
        with torch.no_grad():
            for steps in range(max_steps):

                old_dist = self.environement.GetL2DistFromLandmark(self.scale_state,self.position,self.target)
                X = self.GetState().unsqueeze(0)
                X.to(self.device)
                x = model(X)
                action = x.argmax(dim=1) 
                x = x[0][action]

                self.Move(self.e_greedy(action))
                new_dist = self.environement.GetL2DistFromLandmark(self.scale_state,self.position,self.target)
                reward = old_dist - new_dist
                # print("Reward :",reward)
                reward_lst.append(reward)

                Y = self.GetState().unsqueeze(0)
                Y.to(self.device)
                y = model(Y)
                action = y.argmax(dim=1) 
                y = reward + self.gamma * y[0][action]
                loss = self.loss_fn(y,x)

                steps_loss += loss.item()

        val_loss = steps_loss / max_steps
        print("Loss :", val_loss)
        print("Rewards :", reward_lst)
        print("")


    def PrintBrain(self):
        print("Agent have " + str(len(self.models)) + " DQN with this parameters :")
        print(self.models[0])


def OUT_WARNING():
    print("WARNING : Agent trying to go in a none existing space ")


# e = Environement()
# n = [DRLnet(),DRLnet(),DRLnet()]
# a = TrainingAgent(
#     environement=e,
#     models_lst=n
# )
# print(a.position)