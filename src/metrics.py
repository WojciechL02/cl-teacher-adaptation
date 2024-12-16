from collections import Counter

import numpy as np
import torch


def cka(model1, model2, dataloader, device) -> float:
    model1.eval()
    model2.eval()
    with torch.no_grad():
        cka_list = []
        for images, _ in dataloader:
            images = images.to(device)
            _, features1 = model1(images, return_features=True)
            _, features2 = model2(images, return_features=True)
            _cka = _CKA(features1, features2)
            cka_list.append(_cka)

    return float(sum(cka_list) / len(cka_list))


def _CKA(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return _HSIC(X, Y) / torch.sqrt(_HSIC(X, X) * _HSIC(Y, Y))


def _HSIC(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    GX = X @ X.T
    GY = Y @ Y.T

    n = GX.shape[0]
    H = torch.eye(n, device=X.device) - (1 / n)

    return torch.trace(GX @ H @ GY @ H)


def cm(appr, dataloaders, n_tasks, device):
    confusion_matrix = np.zeros((n_tasks, n_tasks))
    model = appr.model

    model.eval()
    with torch.no_grad():
        for i, dl in enumerate(dataloaders):
            task_ids = []
            for images, targets in dl:
                images = images.to(device)
                targets = targets.to(device)
                outputs, feats = model(images, return_features=True)

                _, _, outputs = appr.classifier.classify(i, outputs, feats, targets, return_dists=True)
                shape = [images.shape[0], len(model.task_cls), model.task_cls[0]]
                curr_data_task_ids = appr.classifier.get_task_ids(outputs, shape)
                task_ids.extend(curr_data_task_ids)

            counts = Counter(task_ids)
            for j, val in counts.items():
                confusion_matrix[i, j] = val / len(dl.dataset)

    return confusion_matrix
