import torch
from torch.utils.data import Dataset, DataLoader
import os
import h5py
import numpy as np
from tqdm import tqdm

class VQVAEDataset(Dataset):
    def __init__(self, filepath, is_test=False, test_ratio=0.1):
        self.filepath = filepath
        self.f = h5py.File(self.filepath, "r", swmr=True, rdcc_nbytes=100*128*128, rdcc_nslots=37)
        self.keys = ['Vx', 'Vy', 'density', 'pressure']
        
        all_handlers = [self.f[key] for key in self.keys]
        
        self.boundaries = np.cumsum([0] + [h.shape[0]*h.shape[1] for h in all_handlers])
        total_length = self.boundaries[-1]

        if is_test:
            self.total_length = int(total_length * test_ratio)
        else: 
            train_ratio = 1 - test_ratio
            self.total_length = int(total_length * train_ratio)

    def __len__(self):
        return int(self.total_length)
    
    def _get_data_handler_and_offset(self, index):
        handler = None  # 初始化handler为None，或者设置为一个默认的值。
        offset_in_dataset = -1

        for i in range(len(self.boundaries)-1):
            if self.boundaries[i] <= (index * len(self)) / self.total_length < self.boundaries[i+1]:
                dataset_index = i  
                adjusted_index = index * len(self) // self.total_length
                offset_in_dataset = adjusted_index - self.boundaries[i]

                handler = self.f[self.keys[dataset_index]]
                break

        if handler is None:
            raise ValueError("Index does not map to any known dataset boundary. Check your boundaries or data indexing logic.")

        return handler, offset_in_dataset
    
    def __getitem__(self, index):
    
        handler, offset_in_dataset = self._get_data_handler_and_offset(index)
        row_idx = offset_in_dataset // handler.shape[1]
        col_idx = offset_in_dataset % handler.shape[1]

        return torch.from_numpy(handler[row_idx, col_idx, :, :].astype(np.float32))
    
    def __del__(self):
        if hasattr(self, 'f'):
            self.f.close()

if __name__ == "__main__":

    filepath = "/mnt/data_4T/chenty/pdebench/2DCFD/2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"  

    train_dataset = VQVAEDataset(filepath)
    test_dataset = VQVAEDataset(filepath, is_test=True)

    print("Train dataset length:", len(train_dataset))
    print("Test dataset length:", len(test_dataset))

    loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=16)
    
    for i, data in enumerate(tqdm(loader)):
        if i == 0: 
            print(data.shape) 


        


