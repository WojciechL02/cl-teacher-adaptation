import torch
import warnings
from argparse import ArgumentParser
from collections import Counter
from copy import deepcopy

from .incremental_learning import Inc_Learning_Appr
from .ft_lb import LieBracketOptimizer
from .loss_func.supcon_loss import SupConLoss
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """Class implementing the Supervised Contrastive Replay approach based on SupCon loss
    described in https://arxiv.org/abs/2004.11362
    """

    def __init__(self, model, device, classifier="nmc", nepochs=60, lr=0.5, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=1e-5, multi_softmax=False, wu_nepochs=0, wu_lr=0, wu_wd=0, wu_fix_bn=False,
                 fix_bn=False, wu_scheduler='constant', wu_patience=None, eval_on_train=False, select_best_model_by_val_loss=True,
                 logger=None, exemplars_dataset=None, scheduler_type=None, temperature=0.1, slca=False, replay_batch_size=100):
        super(Appr, self).__init__(model, device, classifier, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr, wu_fix_bn, wu_scheduler, wu_patience, wu_wd,
                                   fix_bn, eval_on_train, select_best_model_by_val_loss, logger, exemplars_dataset,
                                   scheduler_type, slca=slca)
        if classifier != "nmc":
            raise ValueError("Only NMC is supported in Contrastive Learning approaches.")

        self.loss_func = SupConLoss(temperature)
        self.lamb = 0.3
        self.h1 = 0.05
        self.h2 = 0.15
        self.replay_batch_size = replay_batch_size

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

    def _get_optimizer(self):
        """Returns the optimizer"""
        if self.slca: # and len(self.model.heads) > 1:
            backbone_params = {'params': self.model.model.parameters(), 'lr': self.lr * 0.1}
            head_params = {'params': self.model.heads.parameters()}
            network_params = [backbone_params, head_params]
            return torch.optim.SGD(network_params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

        if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1:
            # if there are no exemplars, previous heads are not modified
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()
        # return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)
        return LieBracketOptimizer(params, self.lr, h=0.01)

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        full_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        if t > 0:
            _ds = deepcopy(self.exemplars_dataset)
            _ds.transform = trn_loader.dataset.transform
            exemplar_loader = torch.utils.data.DataLoader(_ds,
                                                     batch_size=self.replay_batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory
                                                     )
            # print("LENS: ", f"{len(trn_loader)} vs {len(exemplar_loader)}")
            trn_loader = zip(trn_loader, exemplar_loader)

        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()

        for samples in trn_loader:
            images, targets = samples
            if t > 0:
                (images, target), (images_r, target_r) = samples
                bsz_new = target.shape[0]
                bsz_old = target_r.shape[0]
                images_new = torch.cat([images[0], images[1]], dim=0)
                images_old = torch.cat([images_r[0], images_r[1]], dim=0)
                images_new, images_old = images_new.to(self.device), images_old.to(self.device)
                target_new, target_old = target.to(self.device), target_r.to(self.device)
            else:
                bsz_new = targets.shape[0]
                images_new = torch.cat([images[0], images[1]], dim=0)
                images_new, target_new = images_new.to(self.device), targets.to(self.device)
                target_old = None
                features_old = None

            features_new = self.model(images_new)
            f1, f2 = torch.split(features_new, [bsz_new, bsz_new], dim=0)
            features_new = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            if t > 0:
                features_old = self.model(images_old)
                f1, f2 = torch.split(features_old, [bsz_old, bsz_old], dim=0)
                features_old = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            if t > 0:
                loss, loss1, loss2, ngrad1, ngrad2, lie_bracket = self.criterion(t, features_new, target_new, features_old, target_old, is_eval=False)
                lie_bracket_norm = sum(torch.norm(lb) for lb in lie_bracket if lb is not None)
                self.logger.log_scalar(task=None, iter=None, name="loss1", value=loss1.item(), group="train")
                self.logger.log_scalar(task=None, iter=None, name="loss2", value=loss2.item(), group="train")
                self.logger.log_scalar(task=None, iter=None, name="grad1_norm", value=ngrad1.item(), group="train")
                self.logger.log_scalar(task=None, iter=None, name="grad2_norm", value=ngrad2.item(), group="train")
                self.logger.log_scalar(task=t, iter=None, name="lie_bracket_norm", value=lie_bracket_norm.item(), group="train")
            else:
                loss = self.criterion(t, features_new, target_new, features_old, target_old, is_eval=False)
                lie_bracket = None

            self.logger.log_scalar(task=None, iter=None, name="loss", value=loss.item(), group="train")
            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step(lie_bracket)
            # self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        # compute mean of exemplars on every epoch
        # self.classifier.prototypes_update(t, trn_loader, self.val_loader_transform)
        self.classifier.prototypes_update(t, full_loader, self.val_loader_transform)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # add exemplars to train_loader
        # if len(self.exemplars_dataset) > 0 and t > 0:
        #     trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
        #                                              batch_size=trn_loader.batch_size,
        #                                              shuffle=True,
        #                                              num_workers=trn_loader.num_workers,
        #                                              pin_memory=trn_loader.pin_memory)

        super().train_loop(t, trn_loader, val_loader)

        exemplar_selection_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        # UPDATE PROTOTYPES
        self.classifier.prototypes_update(t, exemplar_selection_loader, val_loader.dataset.transform)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, exemplar_selection_loader, val_loader.dataset.transform)

    def eval(self, t, val_loader, log_partial_loss=False):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                images, targets = images.to(self.device), targets.to(self.device)

                _, feats = self.model(images, return_features=True)

                hits_taw, hits_tag = self.classifier.classify(t, None, feats, targets)

                # Log
                total_loss += 0
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def criterion(self, t, outputs_new, targets_new, outputs_old=None, targets_old=None, is_eval=True):
        """Returns the loss value"""

        if is_eval is True:
            return self.loss_func(outputs_new, targets_new)

        elif len(self.exemplars_dataset) > 0:
            L1 = self.loss_func(outputs_old, targets_old)
            L2 = self.loss_func(outputs_new, targets_new)
            # all_outputs = torch.cat([outputs_new, outputs_old], dim=0)
            # all_targets = torch.cat([targets_new, targets_old], dim=0)
            # Lf = self.loss_func(all_outputs, all_targets)

            self.model.zero_grad()

            grad_L1 = torch.autograd.grad(L1, self.model.model.parameters(), create_graph=True)
            grad_L2 = torch.autograd.grad(L2, self.model.model.parameters(), create_graph=True)

            grad_L1_norm = sum(torch.norm(g) for g in grad_L1 if g is not None)
            grad_L2_norm = sum(torch.norm(g) for g in grad_L2 if g is not None)

            hvp_L2_L1 = torch.autograd.grad(grad_L1, self.model.model.parameters(), grad_outputs=grad_L2, retain_graph=True)
            hvp_L1_L2 = torch.autograd.grad(grad_L2, self.model.model.parameters(), grad_outputs=grad_L1, retain_graph=True)

            lie_bracket = [hvp_21 - hvp_12 for hvp_21, hvp_12 in zip(hvp_L2_L1, hvp_L1_L2)]

            self.model.zero_grad()

            L_total = self.lamb * L1 + L2  # + (self.h1 / 4) * grad_L1_norm + (self.h2 / 4) * grad_L2_norm
            return L_total, L1, L2, grad_L1_norm, grad_L2_norm, lie_bracket

        return self.loss_func(outputs_new, targets_new)

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
                    hits_taw, hits_tag, outputs = self.classifier.classify(task_id, None, feats, targets, return_dists=True)

                    # # Log
                    total_acc_tag += hits_tag.sum().data.cpu().numpy().item()
                    total_acc_taw += hits_taw.sum().data.cpu().numpy().item()
                    total_num += len(targets)

                    shape = [images.shape[0], len(self.model.task_cls), self.model.task_cls[0]]
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
