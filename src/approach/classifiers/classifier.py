

class Classifier:
    """Basic class for implementing classifiers for incremental learning approaches"""

    def classify(self, task, outputs, features, targets, return_dists=False):
        pass

    def prototypes_update(self, t, trn_loader, transform):
        pass

    def get_task_ids(self, outputs, stacked_shape):
        pass
