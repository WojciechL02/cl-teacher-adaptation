import torch
import pytest
from unittest.mock import Mock
from approach.classifiers.classifier_factory import ClassifierFactory
from approach.classifiers.linear import LinearClassifier
from approach.classifiers.nmc import NMC
from approach.classifiers.knn import KNN


def test_create_linear():
    clf = ClassifierFactory.create_classifier("linear", torch.device("cpu"), Mock(), Mock())
    assert isinstance(clf, LinearClassifier)


def test_create_nmc():
    clf = ClassifierFactory.create_classifier("nmc", torch.device("cpu"), Mock(), Mock())
    assert isinstance(clf, NMC)


def test_create_knn():
    clf = ClassifierFactory.create_classifier("knn", torch.device("cpu"), Mock(), Mock())
    assert isinstance(clf, KNN)


def test_wrong_argument():
    with pytest.raises(ValueError) as excinfo:
        ClassifierFactory.create_classifier("mlp", torch.device("cpu"), None, None)

    assert str(excinfo.value) == "Unknown classifier type: mlp"
