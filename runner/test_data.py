import os
from glob import glob
from torch.utils.data import Dataset
import torch
import pydicom
import numpy as np
import torchvision.transforms as transforms
from dataset.DcmDataset import *
from testPaths_25_2_10 import testPCPpaths, testLCPpaths, testLCPdelayPaths, testPCPdelayPaths

class DcmDatasetTest(Dataset):
    def __init__(self, 
                datasetRootPath, 
                transforms,
                petDcmRootName="PET WB Corrected_CBM", 
                petDelayDcmRootName="PET WB Corrected_CBM_Delay",
                height=96,
                ):
        super(DcmDatasetTest, self).__init__()
        self.datasetRootPath = datasetRootPath
        self.petDcmRootName = petDcmRootName
        self.petDelayDcmRootName = petDelayDcmRootName
        self.transforms = transforms
        
        self.patient_path_list = testLCPpaths
        print(len(self.patient_path_list))

        self.pet_list = []
        for path in testLCPpaths:
            self.pet_list.append(f"../{path}")

        self.petDelay_list = []
        for path in testLCPdelayPaths:
            self.petDelay_list.append(f"../{path}")
        

        assert len(self.pet_list) == len(self.petDelay_list), "the length of pet and petDelay don't matching!"

    def __getitem__(self, index):
        SUVgap = computeGap(self.pet_list[index])
        # print(type(SUVgap))
        pet = self.transforms.transform(self.pet_list[index])
        petDelay = self.transforms.transform(self.petDelay_list[index])
        delay_time = compute_delay_time(self.pet_list[index], self.petDelay_list[index])

        return {"pet": pet, "petDelay": petDelay, "delay_time": delay_time, "SUVgap": SUVgap}

    def __len__(self):
        return len(self.pet_list)
    