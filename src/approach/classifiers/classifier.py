from typing import Any, Optional


class Classifier:
    """
    Base class for implementing classifiers used in incremental learning approaches.

    This class defines the interface for methods required to perform classification,
    update prototypes, and manage task-specific predictions.
    """

    def classify(
        self,
        task: int,
        outputs: Optional[Any],
        features: Any,
        targets: Any,
        return_dists: bool = False
    ) -> Any:
        """
        Perform classification of input samples.

        Args:
            task (int): The current task index.
            outputs (Optional[Any]): Model outputs, if applicable for the classifier.
            features (Any): Features extracted from the input samples.
            targets (Any): Ground truth labels for the samples.
            return_dists (bool): If True, return distance metrics (if applicable).

        Returns:
            Any: Predicted hits (Task-Aware, Task-Agnostic) and optionally distances.
        """
        pass

    def prototypes_update(
        self,
        t: int,
        trn_loader: Any,
        transform: Any
    ) -> None:
        """
        Update prototypes based on training data for the given task.

        Args:
            t (int): The task index for which to update prototypes.
            trn_loader (Any): DataLoader for the training data.
            transform (Any): Transform to apply to the training data.

        Returns:
            None
        """
        pass

    def get_task_ids(
        self,
        outputs: Any,
        stacked_shape: Any
    ) -> Any:
        """
        Determine task IDs based on model outputs.

        Args:
            outputs (Any): Model outputs for samples.
            stacked_shape (Any): Shape of the stacked outputs.

        Returns:
            Any: List of task IDs corresponding to the outputs.
        """
        pass
