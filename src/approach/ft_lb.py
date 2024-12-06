import torch
from argparse import ArgumentParser
from copy import deepcopy

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset
from torch.optim import Optimizer


class LieBracketOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, h=1e-3):
        """
        Optimizer which includes the Lie Bracket term in the gradient update.

        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
            lr (float): Learning rate.
            h (float): Scaling factor for the Lie Bracket regularization term.
        """
        defaults = dict(lr=lr, h=h)
        super(LieBracketOptimizer, self).__init__(params, defaults)

    def step(self, lie_bracket):
        """
        Performs a single optimization step.

        Args:
            lie_bracket (list[Tensor]): Precomputed Lie Bracket [grad L1, grad L2].
        """
        lie_bracket_len = len(lie_bracket) if lie_bracket is not None else 0
        for group in self.param_groups:
            lr = group['lr']
            h = group['h']

            for i, param in enumerate(group['params']):
                if param.grad is None:
                    continue

                grad = param.grad.data

                if i < lie_bracket_len:
                    lie_bracket_term = lie_bracket[i] * param.data if lie_bracket[i] is not None else 0.0
                    # param.data = (h / 2) * lie_bracket_term - grad
                    # print(param.data)
                else:
                    lie_bracket_term = 0.0
                    # param.data -= lr * grad

                grad = grad.add(lie_bracket_term, alpha=-(h / (2 * lr)))

                # Update rule: θ = θ - lr * (∇L_tilde - h/2 * Lie Bracket)
                param.data -= lr * grad
                


class Appr(Inc_Learning_Appr):
    """Class implementing the finetuning with Lie Bracket"""

    def __init__(self, model, device, classifier="linear", nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr=1e-1, wu_fix_bn=False,
                 wu_scheduler='constant', wu_patience=None, wu_wd=0., fix_bn=False, eval_on_train=False,
                 select_best_model_by_val_loss=True, logger=None, exemplars_dataset=None, scheduler_milestones=False,
                 all_outputs=False, no_learning=False, slca=False):
        super(Appr, self).__init__(model, device, classifier, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr, wu_fix_bn, wu_scheduler, wu_patience, wu_wd,
                                   fix_bn, eval_on_train, select_best_model_by_val_loss, logger, exemplars_dataset,
                                   scheduler_milestones, no_learning, slca=slca)
        self.all_out = all_outputs
        self.lamb = 0.3
        self.h1 = 0.05
        self.h2 = 0.15
        self.replay_batch_size = 100

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--all-outputs', action='store_true', required=False,
                            help='Allow all weights related to all outputs to be modified (default=%(default)s)')
        parser.add_argument('--no-learning', action='store_true', required=False,
                            help='Do not backpropagate gradients after second task - only do batch norm update '
                                 '(default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        if self.slca: # and len(self.model.heads) > 1:
            backbone_params = {'params': self.model.model.parameters(), 'lr': self.lr * 0.1}
            head_params = {'params': self.model.heads.parameters()}
            network_params = [backbone_params, head_params]
            return torch.optim.SGD(network_params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

        if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1 and not self.all_out:
            # if there are no exemplars, previous heads are not modified
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()
        # return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)
        return LieBracketOptimizer(params, self.lr, h=0.2)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # add exemplars to train_loader
        # if len(self.exemplars_dataset) > 0 and t > 0:
        #     trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
        #                                              batch_size=trn_loader.batch_size,
        #                                              shuffle=True,
        #                                              num_workers=trn_loader.num_workers,
        #                                              pin_memory=trn_loader.pin_memory)

        # FINETUNING TRAINING -- contains the epochs loop
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
                images, images_r = images.to(self.device), images_r.to(self.device)
                targets, targets_r = target.to(self.device), target_r.to(self.device)
            else:
                images, targets = images.to(self.device), targets.to(self.device)
                features_old = None
                targets_r = None

            features_new = self.model(images)
            if t > 0:
                features_old = self.model(images_r)

            if t > 0:
                loss, loss1, loss2, ngrad1, ngrad2, lie_bracket = self.criterion(t, features_new, targets, features_old, targets_r, is_eval=False)
                # loss, loss1, loss2, lie_bracket_norm = self.criterion(t, features_new, targets, features_old, targets_r, is_eval=False)
                # loss, loss1, loss2 = self.criterion(t, features_new, targets, features_old, targets_r, is_eval=False)
                lie_bracket_norm = sum(torch.norm(lb) for lb in lie_bracket if lb is not None)
                self.logger.log_scalar(task=None, iter=None, name="loss1", value=loss1.item(), group="train")
                self.logger.log_scalar(task=None, iter=None, name="loss2", value=loss2.item(), group="train")
                self.logger.log_scalar(task=None, iter=None, name="grad1_norm", value=ngrad1.item(), group="train")
                self.logger.log_scalar(task=None, iter=None, name="grad2_norm", value=ngrad2.item(), group="train")
                self.logger.log_scalar(task=None, iter=None, name="lie_bracket_norm", value=lie_bracket_norm.item(), group="train")
            else:
                loss = self.criterion(t, features_new, targets, features_old, targets_r, is_eval=False)
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
        self.classifier.prototypes_update(t, full_loader, self.val_loader_transform)

    def criterion(self, t, outputs_new, targets_new, outputs_old=None, targets_old=None, is_eval=True):
        """Returns the loss value"""

        if is_eval is True:
            return torch.nn.functional.cross_entropy(torch.cat(outputs_new, dim=1), targets_new)

        elif self.all_out or len(self.exemplars_dataset) > 0:
            L1 = torch.nn.functional.cross_entropy(torch.cat(outputs_old[:-1], dim=1), targets_old)
            L2 = torch.nn.functional.cross_entropy(outputs_new[t], targets_new - self.model.task_offset[t])

            self.model.zero_grad()

            grad_L1 = torch.autograd.grad(L1, self.model.model.parameters(), create_graph=True)
            grad_L2 = torch.autograd.grad(L2, self.model.model.parameters(), create_graph=True)

            grad_L1_norm = sum(torch.norm(g) for g in grad_L1 if g is not None)
            grad_L2_norm = sum(torch.norm(g) for g in grad_L2 if g is not None)

            hvp_L2_L1 = torch.autograd.grad(grad_L1, self.model.model.parameters(), grad_outputs=grad_L2, retain_graph=True)
            hvp_L1_L2 = torch.autograd.grad(grad_L2, self.model.model.parameters(), grad_outputs=grad_L1, retain_graph=True)

            lie_bracket = [hvp_21 - hvp_12 for hvp_21, hvp_12 in zip(hvp_L2_L1, hvp_L1_L2)]

            self.model.zero_grad()

            L_total = self.lamb * L1 + L2 + (self.h1 / 4) * grad_L1_norm + (self.h2 / 4) * grad_L2_norm
            return L_total, L1, L2, grad_L1_norm, grad_L2_norm, lie_bracket

        return torch.nn.functional.cross_entropy(outputs_new[t], targets_new - self.model.task_offset[t])

    # def criterion(self, t, outputs_new, targets_new, outputs_old=None, targets_old=None, is_eval=True):
    #     """Returns the loss value"""

    #     if is_eval is True:
    #         return torch.nn.functional.cross_entropy(torch.cat(outputs_new, dim=1), targets_new)

    #     elif self.all_out or len(self.exemplars_dataset) > 0:
    #         L1 = torch.nn.functional.cross_entropy(torch.cat(outputs_old[:-1], dim=1), targets_old)
    #         L2 = torch.nn.functional.cross_entropy(outputs_new[t], targets_new - self.model.task_offset[t])

    #         self.model.zero_grad()

    #         grad_L1 = torch.autograd.grad(L1, self.model.model.parameters(), create_graph=True)
    #         grad_L2 = torch.autograd.grad(L2, self.model.model.parameters(), create_graph=True)

    #         hvp_L2_L1 = torch.autograd.grad(grad_L1, self.model.model.parameters(), grad_outputs=grad_L2, retain_graph=True)
    #         hvp_L1_L2 = torch.autograd.grad(grad_L2, self.model.model.parameters(), grad_outputs=grad_L1, retain_graph=True)

    #         lie_bracket_norm = sum(
    #             torch.norm(hvp_21 - hvp_12) for hvp_21, hvp_12 in zip(hvp_L2_L1, hvp_L1_L2)
    #         )
    #         self.model.zero_grad()

    #         L_total = self.lamb * L1 + L2 + 3e-3 * lie_bracket_norm
    #         return L_total, L1, L2, lie_bracket_norm

    #     return torch.nn.functional.cross_entropy(outputs_new[t], targets_new - self.model.task_offset[t])

    # def criterion(self, t, outputs_new, targets_new, outputs_old=None, targets_old=None, is_eval=True):
    #     """Returns the loss value"""

    #     if is_eval is True:
    #         return torch.nn.functional.cross_entropy(torch.cat(outputs_new, dim=1), targets_new)

    #     elif self.all_out or len(self.exemplars_dataset) > 0:
    #         L1 = torch.nn.functional.cross_entropy(torch.cat(outputs_old[:-1], dim=1), targets_old)
    #         L2 = torch.nn.functional.cross_entropy(outputs_new[t], targets_new - self.model.task_offset[t])
    #         all_outputs = [torch.cat([outputs_new[i], outputs_old[i]], dim=0) for i in range(len(outputs_new))]
    #         all_targets = torch.cat([targets_new, targets_old], dim=0)
    #         Lf = torch.nn.functional.cross_entropy(torch.cat(all_outputs, dim=1), all_targets)

    #         self.model.zero_grad()

    #         grad_L1 = torch.autograd.grad(L1, self.model.model.parameters(), create_graph=True)
    #         grad_L2 = torch.autograd.grad(L2, self.model.model.parameters(), create_graph=True)

    #         grad_L1_norm = sum(torch.norm(g) for g in grad_L1 if g is not None)
    #         grad_L2_norm = sum(torch.norm(g) for g in grad_L2 if g is not None)

    #         hvp_L2_L1 = torch.autograd.grad(grad_L1, self.model.model.parameters(), grad_outputs=grad_L2, retain_graph=True)
    #         hvp_L1_L2 = torch.autograd.grad(grad_L2, self.model.model.parameters(), grad_outputs=grad_L1, retain_graph=True)

    #         lie_bracket = [hvp_21 - hvp_12 for hvp_21, hvp_12 in zip(hvp_L2_L1, hvp_L1_L2)]

    #         self.model.zero_grad()

    #         L_total = Lf + (self.h1 / 4) * grad_L1_norm + (self.h2 / 4) * grad_L2_norm
    #         return L_total, L1, L2, grad_L1_norm, grad_L2_norm, lie_bracket

    #     return torch.nn.functional.cross_entropy(outputs_new[t], targets_new - self.model.task_offset[t])