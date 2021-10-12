
import torch
from torch.utils import data
from Models_class import DRLnet
from Agents_class import (
    DQNAgent,
    RLAgent
)
from Environement_class import Environement

from collections import deque
from sklearn.model_selection import train_test_split

import numpy as np


# ----- MONAI ------
# from monai.losses import DiceCELoss
# from monai.inferers import sliding_window_inference

from monai.transforms import(
    RandShiftIntensityd
)
from monai.data import (
    DataLoader,
    CacheDataset,
    SmartCacheDataset,
    decollate_batch,
)

from GlobalVar import MOVEMENT_ID, MOVEMENT_MATRIX


class TrainingMaster :
    def __init__(
        self,
        environement_lst,
        agent_lst,
        max_memory_size = 100,
        val_percentage = 0.2,
        env_dim = 3,
        num_worker = 1,
        batch_size = 20,
    ) -> None:

        self.environements = environement_lst
        self.env_dim = env_dim
        self.val_percentage = val_percentage
        self.SplitTrainValData(val_percentage)

        self.agents = agent_lst
        self.target_lst = []

        data_dic = {}
        for agent in agent_lst:
            target = agent.target
            self.target_lst.append(target)
            data_dic[target] = {"train" : [], "val" : []}
            for dim in range(self.env_dim):
                data_dic[agent.target]["train"].append(deque(maxlen=max_memory_size))
                data_dic[agent.target]["val"].append(deque(maxlen=int(max_memory_size*val_percentage)))

        self.dataset = data_dic

        self.data_transform = RandShiftIntensityd(keys=["state"],offsets=0.10,prob=0.50,)

        self.num_worker = num_worker
        self.batch_size = batch_size


    # ENVIRONEMENT MANAGEMENT

    def AddEnvironement(self,env):
        self.environements.append(env)

    def ResetEnvironements(self,env_lst = []):
        self.environements = env_lst

    # AGENT MANAGEMENT

    def AddAgent(self,agt):
        self.agents.append(agt)

    def ResetAgents(self,agt_lst = []):
        self.agents = agt_lst


    # DATA MANAGEMENT

    def SplitTrainValData(self,val_percentage = 0.2):
        self.train_env, self.val_env = train_test_split(self.environements, test_size=val_percentage, random_state=len(self.environements))


    def GenerateDataset(self,data_id,data_size):
        data_on_each_env = int(data_size/len(self.train_env)) + 1
        for agent in self.agents:
            for n in range(data_on_each_env):
                for env in self.train_env:
                    for dim in range(self.env_dim):
                        self.dataset[agent.target][data_id][dim].append(env.GetRandomSample(dim,agent.target,agent.start_pos_radius,agent.FOV))
            print(data_id,"dataset generated for Agent :", agent.target)

    def GenerateTrainingDataset(self,data_size):
        self.GenerateDataset("train",data_size)

    def GenerateValidationDataset(self,data_size):
        self.GenerateDataset("val",data_size)

    def GenerateAllDataset(self,data_size):
        self.GenerateDataset("train",data_size)
        self.GenerateDataset("val",int(data_size*self.val_percentage))


    def GenerateDataLoaderDic(self,key):
        data_loader_dic = {}
        for target in self.target_lst:
            ds_data = []
            dl_data = []

            for dim in range(self.env_dim):
                train_ds = CacheDataset(
                    data=self.dataset[target][key][dim],
                    transform=self.data_transform,
                    cache_rate=1.0,
                    num_workers=self.num_worker,
                )

                train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_worker, pin_memory=True)

                ds_data.append(train_ds)
                dl_data.append(train_loader)

            data_loader_dic[target] = {"ds":ds_data,"dl":dl_data}
        return data_loader_dic

    def GenerateDataloaders(self):
        self.GenerateTrainDataLoader()
        self.GenerateValidationDataLoader()

    def GenerateTrainDataLoader(self):
        self.train_data_loaders = self.GenerateDataLoaderDic("train")

    def GenerateValidationDataLoader(self):
        self.val_data_loaders = self.GenerateDataLoaderDic("val")
        
    # TOOLS

    def Train(self,max_epoch,data_size,data_update_freq,val_freq):
        cycle_nbr = int(max_epoch/data_update_freq)
        epoch_ctr = 0
        val_ctr = 0

        for cycle in range(cycle_nbr):
            self.GenerateAllDataset(data_size)
            self.GenerateDataloaders()
            for epoch in range(data_update_freq):
                epoch_ctr +=1
                val_ctr +=1
                print("Epoch :", epoch_ctr,"/",max_epoch)
                # for agent in self.agents:
                    # agent.Train(self.train_data_loaders[agent.target]["dl"])
                    # if val_ctr>=val_freq:
                    #     agent.Validate(self.val_data_loaders[agent.target]["dl"])
                    #     val_ctr = 0
        
        print("End of training")
        for agent in self.agents:
            agent.Validate(self.val_data_loaders[agent.target]["dl"])

    def TrainAgents(self,max_epoch,val_freq):
        epoch = 0
        val_ctr = 0
        while epoch < max_epoch:
            epoch +=1
            val_ctr +=1
            print("Epoch :", epoch,"/",max_epoch)
            for agent in self.agents:
                if val_ctr>=val_freq:
                    agent.Validate(self.val_data_loaders[agent.target]["dl"])
                    val_ctr = 0
                agent.Train(self.train_data_loaders[agent.target]["dl"])

        print("End of training")

        for agent in self.agents:
            agent.Validate(self.val_data_loaders[agent.target]["dl"])
        
        # print(
        # f"train completed, best_metric: {dice_val_best:.4f} "
        # f"at iteration: {global_step_best}"
        # )
        # print("Best model at : ", model_data["best"])
