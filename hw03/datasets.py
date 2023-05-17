from torch.utils.data import Dataset
import os
from PIL import Image

from param import test_tfm

class FoodDataset(Dataset):

    def __init__(self,path,tfm=test_tfm,files = None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files
            
        self.transform = tfm
  
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        
#         plot(im)

        fname = r'{}'.format(fname)
        try:
            label = int(fname.split("/")[-1].split("\\")[-1].split("_")[0])
        except:
            label = -1 # test has no label
            
        return im,label