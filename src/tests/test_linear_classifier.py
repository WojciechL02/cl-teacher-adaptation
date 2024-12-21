import torch
import pytest
from unittest.mock import MagicMock
from approach.classifiers.linear import LinearClassifier


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.task_cls = torch.tensor([5, 10, 15], dtype=torch.int32)
    model.task_offset = torch.tensor([0, 5, 15], dtype=torch.int32)
    model.to.return_value = model
    return model


@pytest.fixture
def mock_exemplars_dataset():
    return MagicMock()


@pytest.fixture
def linear_classifier(mock_model, mock_exemplars_dataset):
    device = torch.device("cpu")
    return LinearClassifier(device, mock_model, mock_exemplars_dataset)


def test_classify_task_aware(linear_classifier):
    task = 1
    outputs = [torch.randn(10, 5), torch.randn(10, 10), torch.randn(10, 15)]
    features = torch.randn(10, 128)
    targets = torch.randint(0, 15, (10,))
    hits_taw, hits_tag = linear_classifier.classify(task, outputs, features, targets, return_dists=False)

    assert hits_taw.shape == (10,)
    assert hits_tag.shape == (10,)


def test_classify_task_agnostic(linear_classifier):
    task = 1
    outputs = [torch.randn(10, 5), torch.randn(10, 10), torch.randn(10, 15)]
    features = torch.randn(10, 128)
    targets = torch.randint(0, 15, (10,))

    hits_taw, hits_tag = linear_classifier.classify(task, outputs, features, targets, return_dists=False)

    assert hits_taw.shape == (10,)
    assert hits_tag.shape == (10,)


def test_classify_with_return_dists(linear_classifier):
    task = 1
    outputs = [torch.randn(10, 5), torch.randn(10, 10), torch.randn(10, 15)]
    features = torch.randn(10, 128)
    targets = torch.randint(0, 15, (10,))

    hits_taw, hits_tag, dists = linear_classifier.classify(task, outputs, features, targets, return_dists=True)

    assert hits_taw.shape == (10,)
    assert hits_tag.shape == (10,)
    assert isinstance(dists, list)
    assert len(dists) == len(outputs)


def test_get_task_ids(linear_classifier):
    outputs = [torch.randn(10, 10), torch.randn(10, 10), torch.randn(10, 10)]
    stacked_shape = [10, 3, 10]

    task_ids = linear_classifier.get_task_ids(outputs, stacked_shape)

    assert len(task_ids) == 10
    assert all(isinstance(task_id, int) for task_id in task_ids)
