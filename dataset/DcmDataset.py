import os
from glob import glob
from torch.utils.data import Dataset
import torch
import pydicom
import numpy as np
import torchvision.transforms as transforms

class DcmDatasetFixed(Dataset):
    def __init__(self,
                datasetRootPath,
                transforms,
                petDcmRootName="PET WB Corrected_CBM", 
                petDelayDcmRootName="PET WB Corrected_CBM_Delay",
                height=96,
                ):
        super().__init__()
        self.datasetRootPath = datasetRootPath
        self.transforms = transforms

        self.pet_list = glob(f"{datasetRootPath}/*/{petDcmRootName}/*")
        self.petDelay_list = glob(f"{datasetRootPath}/*/{petDelayDcmRootName}/*")
        
    def __getitem__(self, index):
        pet = self.transforms.transform(self.pet_list[index])
        petDelay = self.transforms.transform(self.petDelay_list[index])
        delay_time = compute_delay_time(self.pet_list[index], self.petDelay_list[index])

        return {"pet": pet, "petDelay": petDelay, "delay_time": delay_time}
    
    def __len__(self):
        return len(self.pet_list)


class DcmDataset(Dataset):
    def __init__(self, 
                datasetRootPath, 
                transforms,
                petDcmRootName="PET WB Corrected_CBM", 
                petDelayDcmRootName="PET WB Corrected_CBM_Delay",
                height=96,
                ):
        super(DcmDataset, self).__init__()
        self.datasetRootPath = datasetRootPath
        self.petDcmRootName = petDcmRootName
        self.petDelayDcmRootName = petDelayDcmRootName
        self.transforms = transforms
        
        self.patient_path_list = glob(f"{datasetRootPath}/*")
        self.patient_pet_list = []
        self.patient_petDelay_list = []
        for patient_path in self.patient_path_list:
            self.patient_pet_list.append(os.path.join(patient_path, petDcmRootName))
            self.patient_petDelay_list.append(os.path.join(patient_path, petDelayDcmRootName))

        self.pet_list = []
        self.petDelay_list = []
        for patient_pet in self.patient_pet_list:
            patient_dcm_list = glob(f"{patient_pet}/*")
            for patient_dcm in patient_dcm_list:
                self.pet_list.append(patient_dcm)
        for patient_petDelay in self.patient_petDelay_list:
            patient_dcm_list = glob(f"{patient_petDelay}/*")
            for patient_dcm in patient_dcm_list:
                self.petDelay_list.append(patient_dcm)

        assert len(self.pet_list) == len(self.petDelay_list), "the length of pet and petDelay don't matching!"

    def __getitem__(self, index):
        pet = self.transforms.transform(self.pet_list[index])
        petDelay = self.transforms.transform(self.petDelay_list[index])
        delay_time = compute_delay_time(self.pet_list[index], self.petDelay_list[index])

        return {"pet": pet, "petDelay": petDelay, "delay_time": delay_time}

    def __len__(self):
        return len(self.pet_list)
    
def computeGap(path):
    arr = Pet2SUV(path).get_suv()
    return np.max(arr).item() - np.min(arr).item()

class DcmTransforms:
    def __init__(self, resolution=96):
        self.resolution = resolution
        self.transform = transforms.Compose([
            transforms.Lambda(lambda t: Pet2SUV(t).get_suv()),
            transforms.Lambda(lambda t: torch.from_numpy(t).float()),
            transforms.Lambda(lambda t: t.unsqueeze(0)),
            transforms.CenterCrop(self.resolution),
            transforms.Lambda(lambda t: t / (torch.max(t) - torch.min(t))),
            transforms.Lambda(lambda t: (t * 255.).int()),
            transforms.Lambda(lambda t: t / 255.),
            # transforms.Lambda(lambda t: t * 2 - 1),
        ])
        self.reverse_transform = transforms.Compose([
        # transforms.Lambda(lambda t: (t + 1)/ 2),
        transforms.Lambda(lambda t: t.squeeze(0)),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy()),
        ])
    

class Pet2SUV:
    def __init__(self, slice_path):
        self.slice_path = slice_path
        self.ds = pydicom.read_file(slice_path)
        RadiopharmaceuticalInformationSequence = self.ds.RadiopharmaceuticalInformationSequence[0]
        self.AT = self.ds.AcquisitionTime
        self.PW = self.ds.PatientWeight
        self.RST = str(RadiopharmaceuticalInformationSequence['RadiopharmaceuticalStartTime'].value)
        self.RTD = str(RadiopharmaceuticalInformationSequence['RadionuclideTotalDose'].value)
        self.RHL = str(RadiopharmaceuticalInformationSequence['RadionuclideHalfLife'].value)
        self.RS = self.ds.RescaleSlope
        self.RI = self.ds.RescaleIntercept
        self.PET = self.ds.pixel_array

    def dicom_hhmmss_2_s(selof, t):
        t = str(t)
        if len(t) == 5:
            t = "0" + t
        h_t = float(t[0: 2])
        m_t = float(t[2: 4])
        s_t = float(t[4: 6])
        return h_t * 3600 + m_t * 60 + s_t

    def get_suv(self):
        decay_time = self.dicom_hhmmss_2_s(self.RST)
        decay_dose = float(self.RTD) * pow(2, float(decay_time) / float(self.RHL))
        SUVbwScaleFactor = (1000 * float(self.PW)) / decay_dose
        PET_SUV: np.ndarray = (self.PET * float(self.RS) + float(self.RI)) * SUVbwScaleFactor
        return PET_SUV
    
def compute_delay_time(pet_path, petDelay_path):
    pet = pydicom.read_file(pet_path)
    pet_scan_time = pet.StudyTime.split(".")[0]
    pet_scan_hour = int(pet_scan_time[:2])
    pet_scan_min = int(pet_scan_time[2:4])

    petDelay = pydicom.read_file(petDelay_path)
    petDelay_scan_time = petDelay.StudyTime.split(".")[0]
    petDelay_scan_hour = int(petDelay_scan_time[:2])
    petDelay_scan_min = int(pet_scan_time[2:4])

    return (petDelay_scan_hour - pet_scan_hour) * 60 + (petDelay_scan_min - pet_scan_min)

class DcmDatasetCoronal(Dataset):
    def __init__(self):
        super().__init__()
        



# test
if __name__ == "__main__":
    """
    import argparse
    import cv2
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root_path", type=str, default="../../Dual_Time_Dataset/PCP")
    parser.add_argument("--height", type=int, default=96, help="the height of input image")
    args = parser.parse_args()

    DcmTransform = DcmTransforms(resolution=96)
    dataset = DcmDataset(datasetRootPath=args.dataset_root_path, transforms=DcmTransform)
    print(len(dataset))

    for i in range(len(dataset)):
        batch = dataset[i]
        pet = batch["pet"]
        petDelay = batch["petDelay"]
        print(torch.max(pet), torch.min(pet), torch.mean(pet), pet.shape)
        print(torch.max(petDelay), torch.min(petDelay), torch.mean(petDelay), petDelay.shape)
        print("=================")

        cv2.imwrite(f"temp/pet/{i}_pet.png", DcmTransform.reverse_transform(pet))
        cv2.imwrite(f"temp/petDelay/{i}_pet.png", DcmTransform.reverse_transform(petDelay))

    """
    import cv2
    DcmTransform = DcmTransforms(resolution=96)
    dataset = DcmDatasetFixed(datasetRootPath="../../Dual_Time_Dataset/test", transforms=DcmTransform)
    for i in range(len(dataset)):
        batch = dataset[i]
        pet = batch["pet"]
        petDelay = batch["petDelay"]
        print(torch.max(pet), torch.min(pet), torch.mean(pet), pet.shape)
        print(torch.max(petDelay), torch.min(petDelay), torch.mean(petDelay), petDelay.shape)
        print("=================")

        cv2.imwrite(f"temp/pet/{i}_pet.png", DcmTransform.reverse_transform(pet))
        cv2.imwrite(f"temp/petDelay/{i}_pet.png", DcmTransform.reverse_transform(petDelay))
