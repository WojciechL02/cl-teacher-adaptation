import torch
from torch.utils.data import DataLoader
from collections import Counter
from datasets.exemplars_selection import override_dataset_transform


from .classifier import Classifier


class KNN(Classifier):
    """Class implementing the k-Nearest-Neighbors classifier (kNN)"""

    def __init__(self, device, model, exemplars_dataset, k):
        self.device = device
        self.model = model
        self.exemplars_dataset = exemplars_dataset
        self.k = k
        self.data_features = None
        self.data_targets = None

    def _find_classes_from_dists(self, dists):
        out = dists.topk(self.k, largest=False)
        predicted_classes = []
        for i in range(out.indices.shape[0]):
            neighbors_classes = self.data_targets[out.indices[i]]
            class_counts = Counter(neighbors_classes.tolist())
            predicted_classes.append(class_counts.most_common(1)[0][0])
        preds = torch.tensor(predicted_classes).to(self.device)
        return preds

    def classify(self, task, outputs, features, targets, return_dists=False):
        features = features / features.norm(dim=1).view(-1, 1)
        features = features.unsqueeze(1)
        data_features = self.data_features.unsqueeze(0)
        dists = (data_features - features).pow(2).sum(2, keepdim=True).squeeze(dim=2)
        # Task-Agnostic Multi-Head
        preds = self._find_classes_from_dists(dists)
        hits_tag = (preds == targets.to(self.device)).float()

        # Task-Aware Multi-Head
        num_cls = self.model.task_cls[task]
        offset = self.model.task_offset[task]
        indices = torch.where((self.data_targets >= offset) & (self.data_targets < offset + num_cls))[0]
        dists_taw = dists[:, indices]
        preds = self._find_classes_from_dists(dists_taw)
        hits_taw = (preds == targets.to(self.device)).float()
        if return_dists:
            return hits_taw, hits_tag, dists
        return hits_tag, hits_taw

    def prototypes_update(self, t, trn_loader, transform):
        if self.exemplars_dataset._is_active():
            with override_dataset_transform(trn_loader.dataset, transform) as _ds:
                loader = DataLoader(_ds, batch_size=trn_loader.batch_size, shuffle=False,
                                            num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)
                extracted_features = []
                extracted_targets = []
                with torch.no_grad():
                    self.model.eval()
                    for images, targets in loader:
                        feats = self.model(images.to(self.device), return_features=True)[1]
                        # normalize
                        extracted_features.append(feats / feats.norm(dim=1).view(-1, 1))
                        extracted_targets.append(targets)
                self.data_features = torch.cat(extracted_features)
                self.data_targets = torch.cat(extracted_targets).to(self.device)

    def get_task_ids(self, outputs, stacked_shape):
        return [0 for _ in range(stacked_shape[0])]

# if __name__ == "__main__":
#     import torch
#     from torch.utils.data import DataLoader, Dataset

#     class RandomDataset(Dataset):
#         def __init__(self):
#             pass

#         def __len__(self):
#             return 32

#         def __getitem__(self, idx):
#             data = torch.randn((3, 32, 32))
#             label = torch.randint(0, 10, (1,)).item()
#             return data, label

#     dataset = RandomDataset()
#     batch_size = 16
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     clf = KNN(torch.device("cuda:0"), model=None, exemplars_dataset=None, k=3)
#     clf.prototypes_update(0, dataloader, None)
#     clf.classify(0, None, torch.randn((16, 512)), torch.ones((16,)), True)
