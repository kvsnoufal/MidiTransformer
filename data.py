
import torch
import numpy as np
from config import *
class MidiDataset(torch.utils.data.Dataset):
    def __init__(self,df):
        self.df = df
        self.indices = {}
        globalIndex=0
        for fileIndex,row in df.iterrows():
            fname = ENCODING_DIR+"/"+row["fname"]
            with open(fname,'rb') as f:
                nparray = np.load(f)
            for arrayIndex in range(nparray.shape[0]-SEQ_LEN):
                self.indices[globalIndex]={
                    'file':fname,
                    'arrayIndex':arrayIndex
                }
                globalIndex+=1
    def __len__(self):
        return len(self.indices)
    def __getitem__(self,idx):
        fname = self.indices[idx]['file']
        arrayIndex = self.indices[idx]['arrayIndex']
        with open(fname,'rb') as f:
                nparray = np.load(f)
        x = nparray[arrayIndex:arrayIndex+SEQ_LEN]
        y = nparray[arrayIndex+1:arrayIndex+SEQ_LEN+1]

        x = torch.tensor(x,dtype=torch.long)
        y = torch.tensor(y,dtype=torch.long)
        # print(x.shape,y.shape)
        return {"x":x,"y":y}