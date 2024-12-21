import pytest
from unittest.mock import Mock
import torch
from torch.utils.data import DataLoader, TensorDataset
from approach.continual_evaluator import ContinualEvaluator


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.task_cls = [10, 10]

    def forward(self, x, return_features=False):
        batch_size = x.shape[0]
        num_classes = sum(self.task_cls)
        outputs = torch.randn(batch_size, num_classes)
        features = torch.randn(batch_size, 128)
        if return_features:
            return outputs, features
        return outputs

    def eval(self):
        pass


class MockClassifier:
    def classify(self, task_id, outputs, feats, targets, return_dists=False):
        hits_taw = torch.randint(0, 2, (targets.size(0),))
        hits_tag = torch.randint(0, 2, (targets.size(0),))
        if return_dists:
            return hits_taw, hits_tag, outputs
        return hits_taw, hits_tag

    def get_task_ids(self, outputs, shape):
        batch_size = shape[0]
        return [0] * batch_size


class MockLogger:
    def log_scalar(self, task, iter, name, value, group):
        pass


@pytest.fixture
def mock_appr():
    return Mock()


@pytest.fixture
def mock_appr_full():
    mock_appr = Mock()
    mock_appr.device = "cpu"
    mock_appr.model = MockModel()
    mock_appr.classifier = MockClassifier()
    mock_appr.logger = MockLogger()
    return mock_appr


@pytest.fixture
def mock_tst_loader():
    return Mock()


@pytest.fixture
def mock_tst_loader_full():
    task_0_data = TensorDataset(torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,)))
    task_1_data = TensorDataset(torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,)))
    task_2_data = TensorDataset(torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,)))

    return [
        DataLoader(task_0_data, batch_size=4),
        DataLoader(task_1_data, batch_size=4),
        DataLoader(task_2_data, batch_size=4),
    ]


@pytest.fixture
def continual_evaluator(mock_appr, mock_tst_loader):
    return ContinualEvaluator(appr=mock_appr, tst_loader=mock_tst_loader, enabled=True)


@pytest.fixture
def mock_evaluator():
    evaluator = ContinualEvaluator(Mock(), Mock(), enabled=True)
    evaluator.nepochs = 5
    evaluator._log_min_acc = Mock(return_value=0.8)
    evaluator._log_wc_acc = Mock(return_value=0.9)
    evaluator._log_sg_and_rec = Mock()
    return evaluator


def test_prepare_evaluator_sets_nepochs(continual_evaluator):
    t = 3
    nepochs = 10
    continual_evaluator.prepare_evaluator(t, nepochs)
    assert continual_evaluator.nepochs == nepochs


def test_prepare_evaluator_sets_min_accs_prev(continual_evaluator):
    t = 3
    nepochs = 10
    continual_evaluator.prepare_evaluator(t, nepochs)
    assert continual_evaluator.min_accs_prev.shape == (t,)
    assert torch.equal(continual_evaluator.min_accs_prev, torch.ones((t,), requires_grad=False))


def test_prepare_evaluator_does_nothing_if_disabled(mock_appr, mock_tst_loader):
    evaluator = ContinualEvaluator(appr=mock_appr, tst_loader=mock_tst_loader, enabled=False)
    t = 3
    nepochs = 10
    evaluator.prepare_evaluator(t, nepochs)
    assert evaluator.nepochs is None
    assert evaluator.min_accs_prev is None


def test_compute_metrics_task_0_last_epoch(mock_evaluator):
    t = 0
    epoch = 4
    current_acc = 0.75
    prev_t_accs = torch.tensor([])

    mock_evaluator._compute_metrics(t, epoch, prev_t_accs, current_acc)

    assert torch.equal(mock_evaluator.last_e_accs, torch.tensor([current_acc]))
    mock_evaluator._log_min_acc.assert_not_called()
    mock_evaluator._log_wc_acc.assert_not_called()
    mock_evaluator._log_sg_and_rec.assert_not_called()


def test_compute_metrics_task_gt_0_not_last_epoch(mock_evaluator):
    t = 2
    epoch = 2
    current_acc = 0.85
    prev_t_accs = torch.tensor([0.7, 0.8])

    mock_evaluator._compute_metrics(t, epoch, prev_t_accs, current_acc)

    mock_evaluator._log_min_acc.assert_called_once_with(prev_t_accs)
    mock_evaluator._log_wc_acc.assert_called_once_with(t, current_acc, 0.8)
    mock_evaluator._log_sg_and_rec.assert_not_called()
    assert mock_evaluator.last_e_accs is None


def test_compute_metrics_task_gt_0_last_epoch(mock_evaluator):
    t = 3
    epoch = 4
    current_acc = 0.9
    prev_t_accs = torch.tensor([0.65, 0.7, 0.8])

    mock_evaluator._compute_metrics(t, epoch, prev_t_accs, current_acc)

    mock_evaluator._log_min_acc.assert_called_once_with(prev_t_accs)
    mock_evaluator._log_wc_acc.assert_called_once_with(t, current_acc, 0.8)
    mock_evaluator._log_sg_and_rec.assert_called_once_with(prev_t_accs)
    expected_last_e_accs = torch.cat((prev_t_accs, torch.tensor([current_acc])))
    assert torch.equal(mock_evaluator.last_e_accs, expected_last_e_accs)


@pytest.fixture
def mock_evaluator_2():
    evaluator = ContinualEvaluator(Mock(), Mock(), enabled=True)
    evaluator.appr.logger = Mock()
    evaluator.min_accs_prev = torch.tensor([1.0, 1.0, 1.0])
    return evaluator


def test_log_min_acc(mock_evaluator_2):
    prev_t_accs = torch.tensor([0.8, 0.9, 0.7])
    min_acc = mock_evaluator_2._log_min_acc(prev_t_accs)

    expected_min_accs = torch.tensor([0.8, 0.9, 0.7])
    assert torch.equal(mock_evaluator_2.min_accs_prev, expected_min_accs)
    assert min_acc == expected_min_accs.mean().item()


def test_log_wc_acc(mock_evaluator_2):
    t = 2
    current_acc = 0.85
    min_acc = 0.75

    wc_acc = mock_evaluator_2._log_wc_acc(t, current_acc, min_acc)

    k = t + 1
    expected_wc_acc = (1 / k) * current_acc + (1 - (1 / k)) * min_acc
    assert wc_acc == expected_wc_acc


def test_log_wc_acc_zero_current_max_min(mock_evaluator_2):
    t = 2
    current_acc = 0.0
    min_acc = 1.0

    wc_acc = mock_evaluator_2._log_wc_acc(t, current_acc, min_acc)

    k = t + 1
    expected_wc_acc = (1 / k) * current_acc + (1 - (1 / k)) * min_acc
    assert wc_acc == expected_wc_acc


def test_log_wc_acc_current_greater_than_min(mock_evaluator_2):
    t = 3
    current_acc = 0.9
    min_acc = 0.6

    wc_acc = mock_evaluator_2._log_wc_acc(t, current_acc, min_acc)

    k = t + 1
    expected_wc_acc = (1 / k) * current_acc + (1 - (1 / k)) * min_acc
    assert wc_acc == expected_wc_acc


def test_log_wc_acc_high_t_low_current_acc(mock_evaluator_2):
    t = 1000
    current_acc = 0.1
    min_acc = 0.5

    wc_acc = mock_evaluator_2._log_wc_acc(t, current_acc, min_acc)

    k = t + 1
    expected_wc_acc = (1 / k) * current_acc + (1 - (1 / k)) * min_acc
    assert wc_acc == expected_wc_acc


def test_log_wc_acc_equal_current_min(mock_evaluator_2):
    t = 2
    current_acc = 0.75
    min_acc = 0.75

    wc_acc = mock_evaluator_2._log_wc_acc(t, current_acc, min_acc)

    k = t + 1
    expected_wc_acc = (1 / k) * current_acc + (1 - (1 / k)) * min_acc
    assert wc_acc == expected_wc_acc


def test_log_wc_acc_large_t(mock_evaluator_2):
    t = 100
    current_acc = 0.85
    min_acc = 0.75

    wc_acc = mock_evaluator_2._log_wc_acc(t, current_acc, min_acc)

    k = t + 1
    expected_wc_acc = (1 / k) * current_acc + (1 - (1 / k)) * min_acc
    assert wc_acc == expected_wc_acc
