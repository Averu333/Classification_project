from torch.utils.data import Dataset

class MyDataset(Dataset):
    '''
    A custom dataset object to add transformations.
    '''
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset.__getitem__(idx)
        image = data[0]
        image = self.transform(image)
        return (image, data[1])