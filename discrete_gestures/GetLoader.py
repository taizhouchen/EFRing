import torch.utils.data as Data
from EFRingDataLoader import time_to_freq
from sklearn.model_selection import train_test_split
import numpy as np
import torch

class FirstDataset(Data.Dataset):

    def __init__(self,data,label):
        self.data=data
        self.label=label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_index=np.squeeze(time_to_freq(self.data[index]))
        label_index=self.label[index]
        return data_index,label_index

#load the dataset
def get_loader(data,label,batch_size=16,pin_memory=True,divide=False,test_size=0.5):

    dataset=FirstDataset(data,label)
    if divide:
        val_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=test_size,random_state=22)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)

        val_loader=Data.DataLoader(dataset=val_dataset,batch_size=batch_size,pin_memory=pin_memory,num_workers=4)
        test_loader=Data.DataLoader(dataset=test_dataset,batch_size=batch_size,pin_memory=pin_memory,num_workers=4)

        return val_loader,test_loader

    else:
        loader=Data.DataLoader(dataset=dataset,batch_size=batch_size,pin_memory=pin_memory,num_workers=4)

        return loader
