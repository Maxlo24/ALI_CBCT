
import torch
from torch._C import dtype
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
    RandShiftIntensityd,
    Compose,
    ScaleIntensityd
)
from monai.data import (
    DataLoader,
    CacheDataset,
    SmartCacheDataset,
    decollate_batch,
)

class TrainingMaster :
    def __init__(
        self,
        environement_lst,
        agent_lst,
        max_train_memory_size = 100,
        max_val_memory_size = 100,
        val_percentage = 0.2,
        env_dim = 3,
        num_worker = 1,
        batch_size = 20,
    ) -> None:

        self.environements = environement_lst
        # self.env_dim = env_dim
        # self.val_percentage = val_percentage
        self.SplitTrainValData(val_percentage)

        self.agents = agent_lst
        self.target_lst = []

        data_dic = {}
        for agent in agent_lst:
            target = agent.target
            self.target_lst.append(target)
            data_dic[target] = {"train" : [], "val" : []}
            for dim in range(self.env_dim):
                data_dic[agent.target]["train"].append(deque(maxlen=max_train_memory_size))
                data_dic[agent.target]["val"].append(deque(maxlen=max_val_memory_size))

        self.pos_dataset = data_dic

        # self.data_transform = Compose([ScaleIntensityd(keys=["state"],minv = 0.0, maxv = 1.0,factor = None),RandShiftIntensityd(keys=["state"],offsets=0.10,prob=0.50,)])
        self.data_transform = RandShiftIntensityd(keys=["state"],offsets=0.10,prob=0.50)

        self.num_worker = num_worker
        self.batch_size = batch_size

        self.max_train_memory_size = max_train_memory_size
        self.max_val_memory_size = max_val_memory_size



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
        train_env, val_env = train_test_split(self.environements, test_size=val_percentage, random_state=len(self.environements))
        self.s_env = {"train":train_env,"val":val_env}

    # def GenerateAllDataset(self,data_id,data_size):
    #     data_on_each_env = int(data_size/len(self.train_env)) + 1
    #     for agent in self.agents:
    #         for n in range(data_on_each_env):
    #             for env in self.train_env:
    #                 if env.LandmarkIsPresent(agent.target):
    #                     for dim in range(self.env_dim):
    #                         self.dataset[agent.target][data_id][dim].append(env.GetRandomSample(dim,agent.target,agent.start_pos_radius,agent.FOV,agent.movement_matrix))
    #         print(data_id,"dataset generated for Agent :", agent.target)

    def GenerateDataset(self,key,agent,dim):
        return

    def GenerateTrainingDataset(self,data_size):
        self.GenerateDataset("train",data_size)

    def GenerateValidationDataset(self,data_size):
        self.GenerateDataset("val",data_size)

    def GenerateAllDataset(self,data_size):
        self.GenerateDataset("train",data_size)
        self.GenerateDataset("val",int(data_size*self.val_percentage))


    def GenerateDataLoadersDic(self,key):
        data_loader_dic = {}
        for target in self.target_lst:
            dl_data = []

            for dim in range(self.env_dim):
                data_loader = self.GenerateDataLoader(key,target,dim)
                dl_data.append(data_loader)

            data_loader_dic[target] = dl_data
        return data_loader_dic

    def GenerateDataLoader(self,key,target,dim):
        train_ds = CacheDataset(
            data=self.dataset[target][key][dim],
            transform=self.data_transform,
            cache_rate=1.0,
            num_workers=self.num_worker,
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, num_workers=self.num_worker)
        return train_loader

    def GenerateAllDataloaders(self):
        self.GenerateTrainDataLoaders()
        self.GenerateValidationDataLoaders()

    def GenerateTrainDataLoaders(self):
        self.train_data_loaders = self.GenerateDataLoadersDic("train")

    def GenerateValidationDataLoaders(self):
        self.val_data_loaders = self.GenerateDataLoadersDic("val")

    def GenerateDataCaches(self,key):
        data_cash_dic = {}
        for target in self.target_lst:
            ds_data = []

            for dim in range(self.env_dim):
                train_ds = CacheDataset(
                    data=self.dataset[target][key][dim],
                    transform=self.data_transform,
                    cache_rate=1.0,
                    num_workers=self.num_worker,
                )
                ds_data.append(train_ds)
                self.dataset[target][key][dim].clear()
        self.data_cash_dic[key][target] = ds_data
        
    # TOOLS
    def Train(self,max_epoch,val_freq,data_update_freq,data_update_ratio):
        epoch_ctr = 0
        val_ctr = 0
        self.GenerateTrainingDataset(self.max_train_memory_size)
        self.GenerateValidationDataset(self.max_val_memory_size)
        self.GenerateValidationDataLoaders()
        while epoch_ctr < max_epoch:
            val_done = False
            for agent in self.agents:
                for dim in range(self.env_dim):
                    self.GenerateDataset
                    data_loader = self.GenerateDataLoader("train",agent.target,dim)
                    val = val_ctr
                    for i in range(data_update_freq):
                        val += 1
                        agent.Train(data_loader,dim)
                        if val >= val_freq:
                            agent.Validate(self.val_data_loaders[agent.target][dim],dim)
                            val = 0
                            val_done = True

            val_ctr += data_update_freq
            if val_done:
                val_ctr = 0
            epoch_ctr += data_update_freq
            self.GenerateTrainingDataset(int(self.max_train_memory_size*data_update_ratio))

        print("End of training")
        for agent in self.agents:
            for dim in range(self.env_dim):
                agent.Validate(self.val_data_loaders[agent.target][dim],dim)



    # def Train(self,max_epoch,val_freq,data_update_freq,data_update_ratio):
    #     epoch_ctr = 0
    #     val_ctr = 0
    #     update_ctr = 0
    #     self.GenerateTrainingDataset(self.max_train_memory_size)
    #     self.GenerateValidationDataset(self.max_val_memory_size)
    #     # self.GenerateAllDataloaders()

    #     for epoch in range(max_epoch):
    #         if update_ctr>=data_update_freq:
    #             self.GenerateTrainingDataset(int(self.max_train_memory_size*data_update_ratio))
    #             # self.GenerateTrainDataLoaders()
    #             update_ctr = 0
    #         epoch_ctr +=1
    #         val_ctr +=1
    #         update_ctr += 1
    #         print("Epoch :", epoch_ctr,"/",max_epoch)
    #         for agent in self.agents:
    #             agent.Train(self.train_data_loaders[agent.target]["dl"])
    #             if val_ctr>=val_freq:
    #                 agent.Validate(self.val_data_loaders[agent.target]["dl"])
    #         if val_ctr>=val_freq: val_ctr = 0
        
    #     print("End of training")
    #     for agent in self.agents:
    #         agent.Validate(self.val_data_loaders[agent.target]["dl"])

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
