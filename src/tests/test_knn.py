import torch
import pytest
from unittest.mock import MagicMock
from approach.classifiers.knn import KNN


@pytest.fixture
def mock_knn():
    model = MagicMock()
    model.task_cls = [5, 5]
    model.task_offset = [0, 5]
    device = torch.device("cpu")
    exemplars_dataset = MagicMock()
    return KNN(device, model, exemplars_dataset, k=1)


class MockModel:
    def __init__(self, task_cls, task_offset):
        self.task_cls = task_cls
        self.task_offset = task_offset


def test_knn_single_sample_correct_hit(mock_knn):
    task = 0
    features = torch.tensor([[1.0, 0.0, 0.0]])
    targets = torch.tensor([0])

    mock_knn.data_features = torch.tensor([
        [1.0, 0.0, 0.0],  # Class 0
        [0.0, 1.0, 0.0],  # Class 1
        [0.0, 0.0, 1.0],  # Class 2
    ])
    mock_knn.data_targets = torch.tensor([0, 1, 2])

    hits_taw, hits_tag, dists = mock_knn.classify(task, None, features, targets, return_dists=True)

    assert hits_taw.item() == 1.0
    assert hits_tag.item() == 1.0
    assert dists[0, 0].item() == 0.0


def test_knn_single_sample_incorrect_hit(mock_knn):
    task = 0
    features = torch.tensor([[0.5, 0.5, 0.0]])
    targets = torch.tensor([1])

    mock_knn.data_features = torch.tensor([
        [1.0, 0.0, 0.0],  # Class 0
        [0.0, 1.0, 0.0],  # Class 1
        [0.0, 0.0, 1.0],  # Class 2
    ])
    mock_knn.data_targets = torch.tensor([0, 1, 2])

    hits_taw, hits_tag, dists = mock_knn.classify(task, None, features, targets, return_dists=True)

    assert hits_taw.item() == 0.0
    assert hits_tag.item() == 0.0
    assert dists[0, 1].item() == dists[0, 0].item()


def test_knn_multiple_samples(mock_knn):
    task = 0
    features = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    targets = torch.tensor([0, 1, 2])

    mock_knn.data_features = torch.tensor([
        [1.0, 0.0, 0.0],  # Class 0
        [0.0, 1.0, 0.0],  # Class 1
        [0.0, 0.0, 1.0],  # Class 2
    ])
    mock_knn.data_targets = torch.tensor([0, 1, 2])
    hits_taw, hits_tag, dists = mock_knn.classify(task, None, features, targets, return_dists=True)

    assert torch.all(hits_taw == torch.tensor([1.0, 1.0, 1.0]))
    assert torch.all(hits_tag == torch.tensor([1.0, 1.0, 1.0]))
    assert torch.all(dists.diag() == torch.tensor([0.0, 0.0, 0.0]))


def test_knn_k_greater_than_one(mock_knn):
    task = 0
    mock_knn.k = 2
    features = torch.tensor([[1.0, 0.0, 0.0]])
    targets = torch.tensor([0])

    mock_knn.data_features = torch.tensor([
        [1.0, 0.0, 0.0],  # Class 0
        [0.9, 0.1, 0.0],  # Class 0 (close neighbor)
        [0.0, 1.0, 0.0],  # Class 1
    ])
    mock_knn.data_targets = torch.tensor([0, 0, 1])

    hits_taw, hits_tag, dists = mock_knn.classify(task, None, features, targets, return_dists=True)

    assert hits_taw.item() == 1.0
    assert hits_tag.item() == 1.0
    assert dists[0, 0].item() < dists[0, 2].item()  # Class 0 neighbors closer than class 1


def test_find_classes_from_dists_single_sample(mock_knn):
    mock_knn.k = 2
    dists = torch.tensor([[0.1, 0.2, 0.3]])
    classes = torch.tensor([0, 1, 2])

    preds = mock_knn._find_classes_from_dists(dists, classes)

    assert preds.shape == (1,)
    assert preds[0].item() == 0


def test_find_classes_from_dists_multiple_samples(mock_knn):
    mock_knn.k = 2
    dists = torch.tensor([
        [0.1, 0.2, 0.5],
        [0.4, 0.1, 0.2],
    ])
    classes = torch.tensor([0, 1, 2])

    preds = mock_knn._find_classes_from_dists(dists, classes)

    assert preds.shape == (2,)
    assert preds[0].item() == 0
    assert preds[1].item() == 1


def test_find_classes_from_dists_k_greater_than_neighbors(mock_knn):
    mock_knn.k = 5
    dists = torch.tensor([[0.1, 0.2, 0.3]])
    classes = torch.tensor([0, 1, 2])

    preds = mock_knn._find_classes_from_dists(dists, classes)

    assert preds.shape == (1,)
    assert preds[0].item() == 0


def test_get_task_ids_single_task(mock_knn):
    mock_knn.model.task_offset = [0]
    mock_knn.model.task_cls = [5]

    outputs = torch.tensor([[0.1, 0.3, 0.2]])
    mock_knn.data_targets = torch.tensor([0, 1, 2, 3, 4])

    stacked_shape = (1, 5, 1)

    task_ids = mock_knn.get_task_ids(outputs, stacked_shape)

    assert len(task_ids) == 1
    assert task_ids[0] == 0


def test_balance_reference_dataset():
    device = torch.device("cpu")
    task_cls = [5, 1]
    task_offset = [0, 5]
    mock_model = MockModel(task_cls, task_offset)

    knn = KNN(device, mock_model, None, k=3)

    knn.data_features = torch.randn(100, 10)
    knn.data_targets = torch.tensor([0, 1, 2, 3, 4] * 10 + [5] * 50)
    unique_targets, counts = knn.data_targets.unique(sorted=True, return_counts=True)

    t = 1
    knn._balance_reference_dataset(t)

    unique_targets, counts = knn.data_targets.unique(sorted=True, return_counts=True)
    assert all(count == counts[0] for count in counts), "Classes are not balanced after balancing."
    assert len(knn.data_features) == len(knn.data_targets), "Mismatch between features and targets."


def test_balance_reference_dataset_tasks():
    device = torch.device("cpu")
    task_cls = [5, 2]
    task_offset = [0, 5]
    mock_model = MockModel(task_cls, task_offset)

    knn = KNN(device, mock_model, None, k=3)

    knn.data_features = torch.randn(100, 10)
    knn.data_targets = torch.tensor([0, 1, 2, 3, 4] * 10 + [5, 6] * 25)
    _, counts_before = knn.data_targets.unique(sorted=True, return_counts=True)

    t = 1
    knn._balance_reference_dataset(t)

    _, counts_after = knn.data_targets.unique(sorted=True, return_counts=True)
    assert counts_after[-2] < counts_before[-2]
    assert counts_after[-2] < 1.5 * counts_before[0]
    assert counts_after[-1] < counts_before[-1]
    assert counts_after[-1] < 1.5 * counts_before[0]
