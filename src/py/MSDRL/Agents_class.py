from collections import deque

import numpy as np
from scipy.sparse.construct import random

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn

from GlobalVar import*
# from utils import(

# )

from Models_class import DRLnet

def OUT_WARNING():
    print("WARNING : Agent trying to go in a none existing space ")

# #####################################
#  Agents
# #####################################


class DQNAgent :
    def __init__(
        self,
        targeted_landmark,
        movements,
        env_dim,
        brain = None,
        environement = None,
        FOV = [32,32,32],
        start_pos_radius = 20,
        shortmem_size = 10,
        verbose = False
    ) -> None:
    
        self.scale_state = 0
        self.environement = environement
        self.target = targeted_landmark
        self.start_pos_radius = start_pos_radius
        self.start_position = np.array([0,0,0], dtype=np.int16)
        self.position = np.array([0,0,0], dtype=np.int16)
        self.FOV = np.array(FOV, dtype=np.int16)
        self.movement_matrix = movements["mat"]
        self.movement_id = movements["id"]

        self.brain = brain

        self.verbose = verbose

        self.env_dim = env_dim

        position_mem = []
        position_shortmem = []
        for i in range(env_dim):
            position_mem.append([])
            position_shortmem.append(deque(maxlen=shortmem_size))
        self.position_mem = position_mem
        self.position_shortmem = position_shortmem

        self.search_atempt = 0


    def SetEnvironement(self, environement): self.environement = environement

    def SetBrain(self,brain): self.brain = brain

    def ClearShortMem(self):
        for mem in self.position_shortmem:
            mem.clear()

    def GoToScale(self,scale=0):
        self.position = (self.position*(self.environement.GetSpacing(self.scale_state)/self.environement.GetSpacing(scale))).astype(np.int16)
        self.scale_state = scale
        self.search_atempt = 0

    def SetRandomPos(self):
        if self.scale_state == 0:
            rand_coord = np.random.randint(1, self.environement.GetSize(self.scale_state), dtype=np.int16)
            self.start_position = rand_coord
            # rand_coord = self.environement.GetLandmarkPos(self.scale_state,self.target)
        else:
            rand_coord = np.random.randint([1,1,1], self.start_pos_radius*2) - self.start_pos_radius
            rand_coord = self.start_position + rand_coord
            rand_coord = np.where(rand_coord<0, 0, rand_coord)
            rand_coord = rand_coord.astype(np.int16)

        self.position = rand_coord


    def GetState(self):
        state = self.environement.GetZone(self.scale_state,self.position,self.FOV)
        return state

    def UpScale(self):
        scale_changed = False
        if self.scale_state < self.environement.dim -1 :
            self.GoToScale(self.scale_state + 1)
            scale_changed = True
            self.start_position = self.position
        # else:
        #     OUT_WARNING()
        return scale_changed

    def PredictAction(self):
        return self.brain.Predict(self.scale_state,self.GetState())
        
    def Move(self, movement_idx):
        new_pos = self.position + self.movement_matrix[movement_idx]
        if new_pos.all() > 0 and (new_pos < self.environement.GetSize(self.scale_state)).all():
            self.position = new_pos
            # if self.verbose:
            #     print("Moving ", self.movement_id[movement_idx])
        else:
            OUT_WARNING()
            self.ClearShortMem()
            self.SetRandomPos()
            self.search_atempt +=1

    def Train(self, data, dim):
        if self.verbose:
            print("Training agent :", self.target)
        self.brain.Train(data,dim)

    def Validate(self, data,dim):
        if self.verbose:
            print("Validating agent :", self.target)
        self.brain.Validate(data,dim)

    def SavePos(self):
        self.position_mem[self.scale_state].append(self.position)
        self.position_shortmem[self.scale_state].append(self.position)

    def Search(self):
        if self.verbose:
            print("Searching landmark :",self.target)
        self.GoToScale()
        self.SetRandomPos()
        self.SavePos()
        found = False
        while not found:
            # action = self.environement.GetBestMove(self.scale_state,self.position,self.target)
            action = self.PredictAction()
            self.Move(action)
            if self.Visited():
                found = True
            self.SavePos()
            if found:
                if self.verbose:
                    print("Landmark found at scale :",self.scale_state)
                    print("Agent pos = ", self.position, "Landmark pos = ", self.environement.GetLandmarkPos(self.scale_state,self.target))
                scale_changed = self.UpScale()
                found = not scale_changed
            if self.search_atempt > 2:
                print(self.target, "landmark not found")
                self.search_atempt = 0
                return -1

        print("Result :", self.position)
        self.environement.AddPredictedLandmark(self.target,self.position)

    def Visited(self):
        visited = False
        # print(self.position, self.position_shortmem[self.scale_state],)
        for previous_pos in self.position_shortmem[self.scale_state]:
            if np.array_equal(self.position,previous_pos):
                visited = True
        return visited


class RLAgent :
    def __init__(
        self,
        targeted_landmark,
        models = DRLnet,
        FOV = [32,32,32],
        gamma = 0.9,
        epsilon = 0.01,
        lr = 0.0005,
        batch_size = 10,
        max_mem_size = 1000,
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
            models : Type of network to train on each scale,
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
        # self.max_mem_size = max_mem_size
        self.exp_eps = 1.0
        self.exp_end = exp_end
        self.exp_dec = exp_dec
        self.loss_fn = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.models = [
            {"pred" : models(),"target" : models()},
            {"pred" : models(),"target" : models()},
            {"pred" : models(),"target" : models()},
        ]

        self.active_network = self.models[self.scale_state]

        optimizers = []

        for model in self.models:
            model["pred"].to(self.device)
            model["target"].to(self.device)
            optimizer = optim.Adam(model["pred"].parameters(), lr=self.lr)
            optimizers.append(optimizer)
        self.optimizers = optimizers

        # Memory
        self.memory = deque(maxlen=max_mem_size)
        # self.mem_size = max_mem_size
        # self.mem_ctr = 0
        # self.state_mem = np.zeros((self.mem_size, *self.FOV), dtype=np.float32)
        # self.action_mem = np.zeros(self.mem_size, dtype=np.int32)
        # self.reward_mem = np.zeros(self.mem_size, dtype=np.float32)
    def Remember(self,state,action,reward,new_state):
        self.memory.append([state, action, reward, new_state])

    def SetEnvironement(self, environement): self.environement = environement

    def GoToScale(self,scale=0):
        self.position = (self.position*(self.environement.GetSpacing(self.scale_state)/self.environement.GetSpacing(scale))).astype(np.int16)
        self.scale_state = scale
        self.active_network = self.models[self.scale_state]

    def SetRandomPos(self):
        if self.scale_state == 0:
            rand_coord = np.random.randint(1, self.environement.GetSize(self.scale_state), dtype=np.int16)
            # rand_coord = self.environement.GetLandmarkPos(self.scale_state,self.target)
        else:
            rand_coord = np.random.randint([1,1,1], self.start_pos_radius*2) - self.start_pos_radius
            rand_coord = self.environement.GetLandmarkPos(self.scale_state,self.target) #+ rand_coord
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
        rand = np.random.rand(1)
        if rand > self.exp_eps:
            action = wanted_action
        else:
            action = torch.randint(len(self.movement_id),(1,))[0]

        if self.exp_eps > self.exp_end: self.exp_eps-=self.exp_dec
        return action

    def replay(self):
        samples = random.sample(self.memory, self.batch_size)

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


