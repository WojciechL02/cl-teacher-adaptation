import torch
from argparse import ArgumentParser

from .incremental_learning import Inc_Learning_Appr
from .loss_func.supcon_loss import SupConLoss
from .loss_func.arcface_loss import ArcFaceLoss
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """Class implementing the Supervised Contrastive Replay approach based on SupCon loss
    described in https://arxiv.org/abs/2004.11362
    """

    def __init__(self, tst_loader, model, device, classifier="nmc", nepochs=60, lr=0.5, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=1e-5, multi_softmax=False, wu_nepochs=0, wu_lr=0, wu_wd=0, wu_fix_bn=False,
                 fix_bn=False, wu_scheduler='constant', wu_patience=None, eval_on_train=False, select_best_model_by_val_loss=True,
                 logger=None, exemplars_dataset=None, scheduler_type=None, temperature=0.1, loss_func="supcon", slca=False, cont_eval=False, umap_latent=False,
                 log_grad_norm=False, last_head_analysis=False):
        super(Appr, self).__init__(tst_loader, model, device, classifier, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr, wu_fix_bn, wu_scheduler, wu_patience, wu_wd,
                                   fix_bn, eval_on_train, select_best_model_by_val_loss, logger, exemplars_dataset,
                                   scheduler_type, slca=slca, cont_eval=cont_eval, umap_latent=umap_latent, log_grad_norm=log_grad_norm,
                                   last_head_analysis=last_head_analysis)
        if classifier not in ["nmc", "knn"]:
            raise ValueError("Only non-parametric classifiers are supported in Contrastive Learning approaches.")

        if loss_func not in ["supcon", "arcface"]:
            raise ValueError("Only Contrastive Losses can be used in SCR.")
        self.loss_func = SupConLoss(temperature) if loss_func == "supcon" else ArcFaceLoss(temperature)

        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        if not have_exemplars:
            raise ValueError("SCR is expected to use exemplars! Set num-exemplars or num-exemplars-per-class arguments!")

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--temperature', default=0.1, type=float, required=False,
                            help='Temperature coefficient of SupCon loss (default=%(default)s)')
        parser.add_argument('--projector-type', default="mlp", type=str, required=False, choices=["mlp", "linear"],
                            help='Projector type for the contrastive loss based methods (default=%(default)s)')
        parser.add_argument('--loss-func', default="supcon", type=str, required=False, choices=["supcon", "arcface"],
                            help='Type of the Contrastive Loss (default=%(default)s)')
        return parser.parse_known_args(args)

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""

        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()

        for samples in trn_loader:
            images, targets = samples
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

                hits_taw, hits_tag = self.classifier.classify(t, None, feats, targets)

                # Log
                total_loss += 0
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        return self.loss_func(outputs, targets)
