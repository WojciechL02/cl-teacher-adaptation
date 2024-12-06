from .linear import LinearClassifier
from .nmc import NMC
from .knn import KNN


class ClassifierFactory:
    @staticmethod
    def create_classifier(classifier_type, device, model, dataset, best_prototypes=False, multi_softmax=False, k=7):
        if classifier_type == "linear":
            return LinearClassifier(device, model, dataset, multi_softmax)
        elif classifier_type == "nmc":
            return NMC(device, model, dataset, best_prototypes)
        elif classifier_type == "knn":
            return KNN(device, model, dataset, k)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
