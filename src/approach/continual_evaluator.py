import torch
from collections import Counter


class ContinualEvaluator:
    def __init__(self, appr, tst_loader, enabled=True) -> None:
        self._is_active = enabled
        self.appr = appr
        self.tst_loader = tst_loader
        self.nepochs = None
        self.min_accs_prev = None
        self.last_e_accs = None

    def prepare_evaluator(self, t, nepochs):
        if self._is_active:
            self.nepochs = nepochs
            self.min_accs_prev = torch.ones((t,), requires_grad=False)

    def step(self, task, epoch):
        if self._is_active:
            if self.min_accs_prev is None:
                raise RuntimeError("The prepare_evaluator() function must be called at the beggining of task training!")

            prev_t_accs, current_t_acc, _ = self._compute_accs(task)
            self._compute_metrics(task, epoch, prev_t_accs, current_t_acc)

    def _compute_accs(self, t):
        confusion_matrix = torch.zeros((t+1, t+1))
        prev_t_acc = torch.zeros((t,), requires_grad=False)
        current_t_acc = 0.
        sum_acc = 0.
        current_t_acc_taw = 0
        with torch.no_grad():
            loaders = self.tst_loader[:t + 1]

            self.appr.model.eval()
            for task_id, loader in enumerate(loaders):
                total_acc_tag = 0.
                total_acc_taw = 0.
                total_num = 0
                task_ids = []
                for images, targets in loader:
                    images, targets = images.to(self.appr.device), targets.to(self.appr.device)

                    outputs, feats = self.appr.model(images, return_features=True)

                    shape = [images.shape[0], len(self.appr.model.task_cls), self.appr.model.task_cls[0]]
                    hits_taw, hits_tag, outputs = self.appr.classifier.classify(task_id, outputs, feats, targets, return_dists=True)

                    total_acc_tag += hits_tag.sum().data.cpu().numpy().item()
                    total_acc_taw += hits_taw.sum().data.cpu().numpy().item()
                    total_num += len(targets)

                    curr_data_task_ids = self.appr.classifier.get_task_ids(outputs, shape)
                    task_ids.extend(curr_data_task_ids)

                counts = Counter(task_ids)
                for j, val in counts.items():
                    confusion_matrix[task_id, j] = val / len(loader.dataset)

                acc_tag = total_acc_tag / total_num
                acc_taw = total_acc_taw / total_num
                self.appr.logger.log_scalar(task=task_id, iter=None, name="acc_tag", value=100 * acc_tag, group="cont_eval")
                self.appr.logger.log_scalar(task=task_id, iter=None, name="acc_taw", value=100 * acc_taw, group="cont_eval")
                if task_id < t:
                    sum_acc += acc_tag
                    prev_t_acc[task_id] = acc_tag
                else:
                    current_t_acc = acc_tag
                    current_t_acc_taw = acc_taw

        if t > 0:
            recency_bias = confusion_matrix[:-1, -1].mean()
            self.appr.logger.log_scalar(task=None, iter=None, name="task_recency_bias", value=recency_bias.item(), group="cont_eval")
            self.appr.logger.log_scalar(task=None, iter=None, name="avg_acc_tag", value=100 * sum_acc / t, group="cont_eval")

        avg_prev_acc = sum_acc / t if t > 0 else 0.
        return prev_t_acc, current_t_acc, avg_prev_acc

    def _compute_metrics(self, t, epoch, prev_t_accs, current_acc):
        # save accs on last epoch of task 0
        if t == 0 and epoch == self.nepochs - 1:
            self.last_e_accs = torch.tensor([current_acc])

        if t > 0:
            min_acc = self._log_min_acc(prev_t_accs)
            _ = self._log_wc_acc(t, current_acc, min_acc)

            # in last epoch of tasks > 0
            if epoch == self.nepochs - 1:
                self._log_sg_and_rec(prev_t_accs)
                # New last acc of prev tasks (current task becomes a prev task)
                self.last_e_accs = torch.cat((prev_t_accs, torch.tensor([current_acc])))

    def _log_min_acc(self, prev_t_accs):
        self.min_accs_prev = torch.minimum(self.min_accs_prev, prev_t_accs)
        min_acc = self.min_accs_prev.mean().item()
        self.appr.logger.log_scalar(task=None, iter=None, name="min_acc", value=100 * min_acc, group="cont_eval")
        return min_acc

    def _log_wc_acc(self, t, current_acc, min_acc):
        k = t + 1
        wc_acc = (1 / k) * current_acc + (1 - (1 / k)) * min_acc
        self.appr.logger.log_scalar(task=None, iter=None, name="wc_acc", value=100 * wc_acc, group="cont_eval")
        return wc_acc

    def _log_sg_and_rec(self, prev_t_accs):
        # Stability Gap
        sg = self.last_e_accs - self.min_accs_prev
        sg_normalized = torch.div(sg, self.last_e_accs)
        # Recovery
        rec = prev_t_accs - self.min_accs_prev  # in last epoch prev_t_accs is final_acc of prev tasks
        rec_normalized = torch.div(rec, self.last_e_accs)
        # Log SG and REC
        for ts in range(sg.shape[0]):
            self.appr.logger.log_scalar(task=ts, iter=None, name="stability_gap", value=100 * sg[ts].item(), group="cont_eval")
            self.appr.logger.log_scalar(task=ts, iter=None, name="stability_gap_normal", value=100 * sg_normalized[ts].item(), group="cont_eval")

            self.appr.logger.log_scalar(task=ts, iter=None, name="recovery", value=100 * rec[ts].item(), group="cont_eval")
            self.appr.logger.log_scalar(task=ts, iter=None, name="recovery_normal", value=100 * rec_normalized[ts].item(), group="cont_eval")

        self.appr.logger.log_scalar(task=None, iter=None, name="stability_gap_avg", value=100 * sg.mean().item(), group="cont_eval")
        self.appr.logger.log_scalar(task=None, iter=None, name="recovery_avg", value=100 * rec.mean().item(), group="cont_eval")
        self.appr.logger.log_scalar(task=None, iter=None, name="sg_normal_avg", value=100 * sg_normalized.mean().item(), group="cont_eval")
        self.appr.logger.log_scalar(task=None, iter=None, name="recovery_normal_avg", value=100 * rec_normalized.mean().item(), group="cont_eval")


"""
def _continual_evaluation_step(self, t):
        confusion_matrix = torch.zeros((t+1, t+1))
        prev_t_acc = torch.zeros((t,), requires_grad=False)
        current_t_acc = 0.
        sum_acc = 0.
        current_t_acc_taw = 0
        with torch.no_grad():
            loaders = self.tst_loader[:t + 1]

            self.model.eval()
            for task_id, loader in enumerate(loaders):
                total_acc_tag = 0.
                total_acc_taw = 0.
                total_num = 0
                task_ids = []
                for images, targets in loader:
                    images, targets = images.to(self.device), targets.to(self.device)

                    outputs, feats = self.model(images, return_features=True)

                    shape = [images.shape[0], len(self.model.task_cls), self.model.task_cls[0]]
                    hits_taw, hits_tag, outputs = self.classifier.classify(task_id, outputs, feats, targets, return_dists=True)

                    total_acc_tag += hits_tag.sum().data.cpu().numpy().item()
                    total_acc_taw += hits_taw.sum().data.cpu().numpy().item()
                    total_num += len(targets)

                    curr_data_task_ids = self.classifier.get_task_ids(outputs, shape)
                    task_ids.extend(curr_data_task_ids)

                counts = Counter(task_ids)
                for j, val in counts.items():
                    confusion_matrix[task_id, j] = val / len(loader.dataset)

                acc_tag = total_acc_tag / total_num
                acc_taw = total_acc_taw / total_num
                self.logger.log_scalar(task=task_id, iter=None, name="acc_tag", value=100 * acc_tag, group="cont_eval")
                self.logger.log_scalar(task=task_id, iter=None, name="acc_taw", value=100 * acc_taw, group="cont_eval")
                if task_id < t:
                    sum_acc += acc_tag
                    prev_t_acc[task_id] = acc_tag
                else:
                    current_t_acc = acc_tag
                    current_t_acc_taw = acc_taw

            if t > 0:
                # Average accuracy over all previous tasks
                self.logger.log_scalar(task=None, iter=None, name="avg_acc_tag", value=100 * sum_acc / t, group="cont_eval")

        if t > 0:
            recency_bias = confusion_matrix[:-1, -1].mean()
            self.logger.log_scalar(task=None, iter=None, name="task_recency_bias", value=recency_bias.item(), group="cont_eval")

        avg_prev_acc = sum_acc / t if t > 0 else 0.
        return prev_t_acc, current_t_acc, avg_prev_acc
"""