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

    def _find_classes_from_dists(self, dists, classes):
        out = dists.topk(self.k, largest=False, sorted=True)
        predicted_classes = []
        for i in range(out.indices.shape[0]):
            neighbors_classes = classes[out.indices[i]]
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
        preds = self._find_classes_from_dists(dists, self.data_targets)
        hits_tag = (preds == targets.to(self.device)).float()

        # Task-Aware Multi-Head
        num_cls = self.model.task_cls[task]
        offset = self.model.task_offset[task]
        indices = torch.where((self.data_targets >= offset) & (self.data_targets < offset + num_cls))[0]
        dists_taw = dists[:, indices]
        classes_taw = self.data_targets[indices]
        preds = self._find_classes_from_dists(dists_taw, classes_taw)
        hits_taw = (preds == targets.to(self.device)).float()
        if return_dists:
            return hits_taw, hits_tag, dists
        return hits_taw, hits_tag

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

                # BALANCE REFERENCE DATASET
                _, counts = self.data_targets.unique(sorted=True, return_counts=True)
                min_count = counts[0] * self.model.task_cls[t]
                current_task_classes = torch.arange(0, self.model.task_cls[t]) + self.model.task_offset[t]
                indices = torch.nonzero(self.data_targets.unsqueeze(1) == current_task_classes.to(self.device), as_tuple=False)[:, 0]
                balanced_current_task = indices[torch.randperm(indices.size(0))[:min_count]]

                all_indices = torch.arange(self.data_targets.size(0)).to(self.device)
                old_data_indices = all_indices[~torch.isin(all_indices, indices.to(self.device))]
                total_balanced_indices = torch.cat([balanced_current_task, old_data_indices])

                self.data_targets = self.data_targets[total_balanced_indices.to(self.device)]
                self.data_features = self.data_features[total_balanced_indices.to(self.device)]

    def get_task_ids(self, outputs, stacked_shape):
        preds = self._find_classes_from_dists(outputs, self.data_targets)

        task_ranges = [
            (offset, offset + num) for offset, num in zip(self.model.task_offset, self.model.task_cls)
        ]
        task_ids = torch.zeros_like(preds, dtype=torch.long)
        for task_id, (start, end) in enumerate(task_ranges):
            mask = (preds >= start) & (preds < end)
            task_ids[mask] = task_id
        return task_ids.tolist()
