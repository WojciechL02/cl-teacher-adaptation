import time
from copy import deepcopy
from collections import Counter
import torch
import numpy as np
from argparse import ArgumentParser
import umap
import pandas as pd

from loggers.exp_logger import ExperimentLogger
from datasets.exemplars_dataset import ExemplarsDataset

from .classifier import Classifier


class LinearClassifier(Classifier):
    """Basic class for implementing classifiers for incremental learning approaches"""

    def __init__(self, device, model, exemplars_dataset, multi_softmax=False):
        super(LinearClassifier, self).__init__()
        self.device = device
        self.model = model
        self.multi_softmax = multi_softmax
        self.exemplars_dataset = exemplars_dataset

    def classify(self, task, outputs, features, targets, return_dists=False):
        pred = torch.zeros_like(targets)
        # Task-Aware Multi-Head
        for m in range(len(pred)):
            this_task = (self.model.task_cls.cumsum(0).to(self.device) <= targets[m]).sum()
            pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
        hits_taw = (pred == targets).float()
        # Task-Agnostic Multi-Head
        if self.multi_softmax:
            outputs = [torch.nn.functional.log_softmax(output, dim=1) for output in outputs]
            pred = torch.cat(outputs, dim=1).argmax(1)
        else:
            pred = torch.cat(outputs, dim=1).argmax(1)
        hits_tag = (pred == targets).float()
        if return_dists:
            return hits_taw, hits_tag, outputs
        return hits_taw, hits_tag

    def get_task_ids(self, outputs, stacked_shape):
        outputs = torch.stack(outputs, dim=1)
        outputs = torch.max(outputs, dim=-1)[0]
        return outputs.argmax(dim=-1).tolist()
