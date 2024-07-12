from torch.utils.data import Dataset, TensorDataset

class PairedDataset(Dataset):
    def __init__(self, abnormal_dataset, normal_dataset,label):
        self.abnormal_dataset = abnormal_dataset
        self.normal_dataset = normal_dataset
        self.length = min(len(abnormal_dataset), len(normal_dataset))
        self.label = label

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x, _ = self.abnormal_dataset[index]
        y, _ = self.normal_dataset[index]

        return x, y, self.label
