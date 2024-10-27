import torch
import warnings
from argparse import ArgumentParser
from collections import Counter

from .incremental_learning import Inc_Learning_Appr
from .loss_func.supcon_loss import SupConLoss
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """Class implementing the Supervised Contrastive Replay approach based on SupCon loss
    described in https://arxiv.org/abs/2004.11362
    """

    def __init__(self, model, device, classifier="nmc", nepochs=60, lr=0.5, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=1e-5, multi_softmax=False, wu_nepochs=0, wu_lr=0, wu_wd=0, wu_fix_bn=False,
                 fix_bn=False, wu_scheduler='constant', wu_patience=None, eval_on_train=False, select_best_model_by_val_loss=True,
                 logger=None, exemplars_dataset=None, scheduler_milestones=None, temperature=0.1, slca=False):
        super(Appr, self).__init__(model, device, classifier, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr, wu_fix_bn, wu_scheduler, wu_patience, wu_wd,
                                   fix_bn, eval_on_train, select_best_model_by_val_loss, logger, exemplars_dataset,
                                   scheduler_milestones, slca=slca)
        if classifier != "nmc":
            raise ValueError("Only NMC is supported in Contrastive Learning approaches.")

        self.loss_func = SupConLoss(temperature)

        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        if not have_exemplars:
            warnings.warn("Warning: SupCon is expected to use exemplars.")

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--temperature', default=0.1, type=float, required=False,
                            help='Temperature coefficient of SupCon loss (default=%(default)s)')
        return parser.parse_known_args(args)

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()

        for images, targets in trn_loader:
            images = torch.cat([images[0], images[1]], dim=0)
            images, targets = images.to(self.device), targets.to(self.device)
            bsz = targets.shape[0]

            features = self.model(images)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            loss = self.criterion(t, features, targets)
            self.logger.log_scalar(task=None, iter=None, name="loss", value=loss.item(), group="train")
            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        # compute mean of exemplars on every epoch
        self.classifier.prototypes_update(t, trn_loader, self.val_loader_transform)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        super().train_loop(t, trn_loader, val_loader)

        # UPDATE PROTOTYPES
        self.classifier.prototypes_update(t, trn_loader, val_loader.dataset.transform)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def eval(self, t, val_loader, log_partial_loss=False):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                images, targets = images.to(self.device), targets.to(self.device)

                _, feats = self.model(images, return_features=True)

                ### SIWY DYM ###
                hits_taw, hits_tag = self.classifier.classify(t, [], feats, targets)
                # Log
                total_loss += 0
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        return self.loss_func(outputs, targets)

    def _continual_evaluation_step(self, t):
        confusion_matrix = torch.zeros((t+1, t+1))
        prev_t_acc = torch.zeros((t,), requires_grad=False)
        current_t_acc = 0.
        sum_acc = 0.
        total_loss_curr = 0.
        total_num_curr = 0
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
                    # Forward current model
                    _, feats = self.model(images, return_features=True)
                    # features = torch.cat([feats.unsqueeze(1), feats.unsqueeze(1)], dim=1)

                    if task_id == t:
                        # loss = self.criterion(t, features, targets)
                        # total_loss_curr += loss.item() * len(targets)
                        total_loss_curr += 0
                        total_num_curr += total_num

                    # outputs_stacked = torch.stack(outputs, dim=1)
                    # shape = outputs_stacked.shape
                    hits_taw, hits_tag, outputs = self.classifier.classify(task_id, [], feats, targets, return_dists=True)

                    # # Log
                    total_acc_tag += hits_tag.sum().data.cpu().numpy().item()
                    total_acc_taw += hits_taw.sum().data.cpu().numpy().item()
                    total_num += len(targets)

                    ### TO JEST DO ROZWAŻENIA ######################
                    outputs = outputs.view(images.shape[0], len(self.model.task_cls), self.model.task_cls[0])
                    outputs = torch.min(outputs, dim=-1)[0]
                    outputs = outputs.argmin(dim=-1)
                    task_ids.extend(outputs.tolist())
                    ###############################################

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
        return prev_t_acc, current_t_acc, avg_prev_acc  #, total_loss_curr / total_num_curr, current_t_acc_taw
        # acc poprzednich tasków, acc na aktualnym tasku, średnia z poprzednich tasków
