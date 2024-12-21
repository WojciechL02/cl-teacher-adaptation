import torch
import pytest
import numpy as np
from unittest.mock import MagicMock
from approach.classifiers.nmc import NMC


@pytest.fixture
def mock_nmc():
    model = MagicMock()
    model.task_cls = [5, 5]
    model.task_offset = [0, 5]
    device = torch.device("cpu")
    exemplars_dataset = MagicMock()
    return NMC(device, model, exemplars_dataset)


def test_classify_basic(mock_nmc):
    task = 0
    outputs = [torch.rand(2, 10)]
    features = torch.rand(2, 10)
    targets = torch.tensor([1, 4])

    mock_nmc.exemplar_means = [torch.rand(10) for _ in range(10)]

    hits_taw, hits_tag = mock_nmc.classify(task, outputs, features, targets)

    assert hits_taw.shape == torch.Size([2])
    assert hits_tag.shape == torch.Size([2])
    assert hits_taw.dtype == torch.float
    assert hits_tag.dtype == torch.float


def test_classify_single_sample(mock_nmc):
    task = 0
    features = torch.full((1, 10), 3.89)
    targets = torch.tensor([4,])

    mock_nmc.exemplar_means = [torch.full((10,), i) for i in range(10)]

    hits_taw, hits_tag = mock_nmc.classify(task, None, features, targets)
    assert hits_taw.shape == (1,)
    assert hits_tag.shape == (1,)


def test_classify_single_sample_correct_hit(mock_nmc):
    task = 0
    features = torch.tensor([[1.0, 0.0, 0.0]])
    targets = torch.tensor([0])

    mock_nmc.exemplar_means = [
        torch.tensor([1.0, 0.0, 0.0]),  # Class 0
        torch.tensor([0.0, 1.0, 0.0]),  # Class 1
        torch.tensor([0.0, 0.0, 1.0])   # Class 2
    ]

    hits_taw, hits_tag, dists = mock_nmc.classify(task, None, features, targets, return_dists=True)

    assert hits_taw.item() == 1.0
    assert hits_tag.item() == 1.0
    assert dists[0, 0].item() == 0.0  # Distance to correct class is 0


def test_classify_single_sample_incorrect_hit(mock_nmc):
    task = 0
    features = torch.tensor([[0.5, 0.5, 0.0]])
    targets = torch.tensor([1])

    mock_nmc.exemplar_means = [
        torch.tensor([1.0, 0.0, 0.0]),  # Class 0
        torch.tensor([0.0, 1.0, 0.0]),  # Class 1
        torch.tensor([0.0, 0.0, 1.0])   # Class 2
    ]

    hits_taw, hits_tag, dists = mock_nmc.classify(task, None, features, targets, return_dists=True)

    assert hits_taw.item() == 0.0
    assert hits_tag.item() == 0.0
    assert dists[0, 1].item() == dists[0, 0].item()


def test_classify_multiple_samples(mock_nmc):
    task = 0
    features = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    targets = torch.tensor([0, 1, 2])

    mock_nmc.exemplar_means = [
        torch.tensor([1.0, 0.0, 0.0]),  # Class 0
        torch.tensor([0.0, 1.0, 0.0]),  # Class 1
        torch.tensor([0.0, 0.0, 1.0])   # Class 2
    ]

    hits_taw, hits_tag, dists = mock_nmc.classify(task, None, features, targets, return_dists=True)

    assert torch.all(hits_taw == torch.tensor([1.0, 1.0, 1.0]))
    assert torch.all(hits_tag == torch.tensor([1.0, 1.0, 1.0]))
    assert dists[0, 0].item() == 0.0
    assert dists[1, 1].item() == 0.0
    assert dists[2, 2].item() == 0.0


def test_extract_features_and_targets(mock_nmc):
    dataloader = MagicMock()
    dataloader.__iter__.return_value = [
        (torch.rand(4, 3, 32, 32), torch.tensor([0, 1, 2, 3])),
        (torch.rand(4, 3, 32, 32), torch.tensor([4, 5, 6, 7]))
    ]

    mock_nmc.model.return_value = (None, torch.rand(4, 10))

    features, targets = mock_nmc._extract_features_and_targets(dataloader)

    assert features.shape[0] == 8
    assert len(targets) == 8


def test_compute_mean_of_exemplars(mock_nmc):
    transform = MagicMock()
    trn_loader = MagicMock()
    trn_loader.batch_size = 32
    trn_loader.num_workers = 0
    trn_loader.pin_memory = False

    mock_nmc.exemplars_dataset = MagicMock()
    mock_nmc.exemplars_dataset.__len__.return_value = 100

    mock_nmc._extract_features_and_targets = MagicMock(
        return_value=(torch.rand(100, 10), np.random.randint(0, 5, size=100))
    )

    mock_nmc.compute_mean_of_exemplars(trn_loader, transform)

    assert len(mock_nmc.exemplar_means) > 0
    for mean in mock_nmc.exemplar_means:
        assert torch.is_tensor(mean)
        assert mean.shape[0] == 10


def test_prototypes_update(mock_nmc):
    t = 1
    trn_loader = MagicMock()
    transform = MagicMock()

    mock_nmc.exemplars_dataset._is_active = MagicMock(return_value=True)
    mock_nmc.compute_mean_of_exemplars = MagicMock()
    mock_nmc.compute_means_of_current_classes = MagicMock()

    mock_nmc.prototypes_update(t, trn_loader, transform)

    mock_nmc.compute_mean_of_exemplars.assert_called_once_with(trn_loader, transform)
    mock_nmc.compute_means_of_current_classes.assert_called_once_with(trn_loader, transform)


def test_get_task_ids(mock_nmc):
    outputs = torch.rand(8, 2, 5)
    stacked_shape = (8, 2, 5)

    task_ids = mock_nmc.get_task_ids(outputs, stacked_shape)

    assert len(task_ids) == 8
    assert all(0 <= tid < 2 for tid in task_ids)
