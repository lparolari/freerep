import torch


class SubsetDataset(torch.utils.data.Dataset):
    """
    Subset of a dataset at specified indices.
    
    Compared to ``torch.utils.data.Subset``, this implementation additionally
    supports a coco-like interface.
    """

    def __init__(self, dataset, indices):
        self.ds = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self.ds[self.indices[i]] for i in idx]
        return self.ds[self.indices[idx]]

    def id(self, index):
        return self.ds.id(self.indices[index])

    def img(self, id):
        return self.ds.img(id)
    
    def img_ann(self, id):
        return self.ds.img_ann(id)
    
    def tgt_ann(self, id):
        return self.ds.tgt_ann(id)
