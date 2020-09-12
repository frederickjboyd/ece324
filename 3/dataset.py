import torch.utils.data as data


class AdultDataset(data.Dataset):
    def __init__(self, features, label):
        self.features = features
        self.label = label

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        features = self.features[index]
        label = self.label[index]

        return features, label
