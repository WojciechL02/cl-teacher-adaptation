import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.exemplars_selection import override_dataset_transform

from .classifier import Classifier


class NMC(Classifier):
    """Class implementing the Nearest-Mean-Classifier (NMC)"""

    def __init__(self, device, model, exemplars_dataset, best_prototypes=False):
        self.device = device
        self.model = model
        self.exemplars_dataset = exemplars_dataset
        self.exemplar_means = []
        self.previous_datasets = []
        self.best_prototypes = best_prototypes

    def classify(self, task, outputs, features, targets, return_dists=False):
        # expand means to all batch images
        means = torch.stack(self.exemplar_means)
        means = torch.stack([means] * features.shape[0])
        means = means.transpose(1, 2)
        # expand all features to all classes
        features = features / features.norm(dim=1).view(-1, 1)
        features = features.unsqueeze(2)
        features = features.expand_as(means)
        # get distances for all images to all exemplar class means -- nearest prototype
        dists = (features - means).pow(2).sum(1, keepdim=True).squeeze(dim=1)
        # Task-Aware Multi-Head
        num_cls = self.model.task_cls[task]
        offset = self.model.task_offset[task]
        pred = dists[:, offset:offset + num_cls].argmin(1)
        hits_taw = (pred + offset == targets.to(self.device)).float()
        # Task-Agnostic Multi-Head
        pred = dists.argmin(1)
        hits_tag = (pred == targets.to(self.device)).float()
        if return_dists:
            return hits_taw, hits_tag, dists
        return hits_taw, hits_tag

    def _extract_features_and_targets(self, dataloader):
        extracted_features = []
        extracted_targets = []
        with torch.no_grad():
            self.model.eval()
            for images, targets in dataloader:
                feats = self.model(images.to(self.device), return_features=True)[1]
                # normalize
                extracted_features.append(feats / feats.norm(dim=1).view(-1, 1))
                extracted_targets.extend(targets)
        extracted_features = torch.cat(extracted_features)
        extracted_targets = np.array(extracted_targets)
        return extracted_features, extracted_targets

    def compute_mean_of_exemplars(self, trn_loader, transform):
        dataset = self.previous_datasets[0] if self.best_prototypes else self.exemplars_dataset
        if self.best_prototypes:
            if len(self.previous_datasets) > 1:
                for subset in self.previous_datasets[1:-1]:
                    dataset += subset
        with override_dataset_transform(dataset, transform) as _ds:
            icarl_loader = DataLoader(_ds, batch_size=trn_loader.batch_size, shuffle=False,
                                      num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)
            extracted_features, extracted_targets = self._extract_features_and_targets(icarl_loader)
            for curr_cls in np.unique(extracted_targets):
                cls_ind = np.where(extracted_targets == curr_cls)[0]
                cls_feats = extracted_features[cls_ind]
                cls_feats_mean = cls_feats.mean(0) / cls_feats.mean(0).norm()
                self.exemplar_means.append(cls_feats_mean)

    def compute_means_of_current_classes(self, loader, transform):
        with override_dataset_transform(loader.dataset, transform) as _ds:
            icarl_loader = DataLoader(_ds, batch_size=loader.batch_size, shuffle=False,
                                    num_workers=loader.num_workers, pin_memory=loader.pin_memory)
            extracted_features, extracted_targets = self._extract_features_and_targets(icarl_loader)
            for curr_cls in np.unique(extracted_targets):
                if curr_cls >= self.model.task_offset[-1]:
                    cls_ind = np.where(extracted_targets == curr_cls)[0]
                    cls_feats = extracted_features[cls_ind]
                    cls_feats_mean = cls_feats.mean(0) / cls_feats.mean(0).norm()
                    self.exemplar_means.append(cls_feats_mean)

    def prototypes_update(self, t, trn_loader, transform):
        if self.exemplars_dataset._is_active():
            self.exemplar_means = []
            if t > 0:
                self.compute_mean_of_exemplars(trn_loader, transform)
            self.compute_means_of_current_classes(trn_loader, transform)

    def get_task_ids(self, outputs, stacked_shape):
        outputs = outputs.view(stacked_shape[0], stacked_shape[1], stacked_shape[2])
        outputs = torch.min(outputs, dim=-1)[0]
        outputs = outputs.argmin(dim=-1)
        return outputs.tolist()
