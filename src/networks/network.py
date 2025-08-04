import torch
from torch import nn
from copy import deepcopy
from networks.projectors import LowRankProjector, FullProjector, DifferentialProjector, MLPProjector, DoubleProjector


class LLL_Net(nn.Module):
    """Basic class for implementing networks"""

    def __init__(
        self,
        model,
        remove_existing_head=False,
        head_init_mode=None,
        projector_type=None,
    ):
        head_var = model.head_var
        assert type(head_var) == str
        assert not remove_existing_head or hasattr(
            model, head_var
        ), "Given model does not have a variable called {}".format(head_var)
        assert not remove_existing_head or type(getattr(model, head_var)) in [
            nn.Sequential,
            nn.Linear,
        ], "Given model's head {} does is not an instance of nn.Sequential or nn.Linear".format(
            head_var
        )
        super(LLL_Net, self).__init__()

        self.model = model
        self.head_init_mode = head_init_mode
        last_layer = getattr(self.model, head_var)

        if remove_existing_head:
            if type(last_layer) == nn.Sequential:
                self.out_size = last_layer[-1].in_features
                # strips off last linear layer of classifier
                del last_layer[-1]
            elif type(last_layer) == nn.Linear:
                self.out_size = last_layer.in_features
                # converts last layer into identity
                # setattr(self.model, head_var, nn.Identity())
                # WARNING: this is for when pytorch version is <1.2
                setattr(self.model, head_var, nn.Sequential())
        else:
            self.out_size = last_layer.out_features

        self.primary_state_dict = deepcopy(self.model.state_dict())

        self.heads = nn.ModuleList()

        self.ln = nn.LayerNorm(self.out_size)
        self.ln2 = nn.LayerNorm(self.out_size)
        self.projector_type = projector_type
        if projector_type is not None:
            if projector_type == "full":
                self.projector = FullProjector(dim=self.out_size)
            elif projector_type == "mlp":
                self.projector = MLPProjector(dim=self.out_size)
            elif projector_type == "low_rank":
                self.projector = LowRankProjector(dim=self.out_size, rank=32)
            elif projector_type == "differential":
                self.projector = DifferentialProjector(dim=self.out_size, rank=32)
            elif projector_type == "double":
                self.projector = DoubleProjector(dim=self.out_size)
                self.ln3 = nn.LayerNorm(self.out_size)

        self.task_cls = []
        self.task_offset = []
        self._initialize_weights()

    def add_head(self, num_outputs):
        """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the
        corresponding offsets
        """
        self.heads.append(nn.Linear(self.out_size, num_outputs))
        # first head has the same init (zeros) in all methods
        if len(self.heads) == 1:
            nn.init.zeros_(self.heads[-1].weight)
            nn.init.zeros_(self.heads[-1].bias)

        # weights initialization for other heads
        elif self.head_init_mode is not None:
            self._initialize_head_weights()

        # Projector reinit
        if self.projector_type is not None:
            self.projector.reset_weights()

        # we re-compute instead of append in case an approach makes changes to the heads
        self.task_cls = torch.tensor([head.out_features for head in self.heads])
        self.task_offset = torch.cat(
            [torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]]
        )

    def forward(
        self, x, return_features=False, is_eval=False, return_proj_output=False
    ):
        """Applies the forward pass

        Simplification to work on multi-head only -- returns all head outputs in a list
        Args:
            x (tensor): input images
            return_features (bool): return the representations before the heads
        """
        x = self.model(x)
        if self.projector_type is not None and self.projector_type != "double":
            x = self.ln(x)
            x_1 = self.projector(x)
            # x_ = x_1
            x_ = self.ln2(x_1 + x)
        elif self.projector_type == "double":
            x = self.ln(x)
            x_1, x_2 = self.projector(x)
            x_proj1 = self.ln2(x_1) + x
            x_proj2 = self.ln3(x_2) + x

        assert len(self.heads) > 0, "Cannot access any head"

        y = []
        for head_id, head in enumerate(self.heads):
            if (not is_eval) and self.projector_type is not None:
                if self.projector_type == "double":
                    if head_id < len(self.heads) - 1:
                        y.append(head(x_proj1))
                    else:
                        y.append(head(x_proj2))
                else:
                    y.append(head(x_))
            else:
                y.append(head(x))
        if return_features:
            if self.projector_type is not None and return_proj_output:
                if self.projector_type == "double":
                    return y, x, x_1, x_2, x_proj1, x_proj2
                return y, x, x_1, x_
            else:
                return y, x
        else:
            return y

    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return

    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False

    def freeze_backbone(self):
        """Freeze all parameters from the main model, but not the heads"""
        for param in self.model.parameters():
            param.requires_grad = False

    def reset_backbone(self):
        """Reset all parameters from the main model, but not the heads"""
        self.model.load_state_dict(self.primary_state_dict)

    def freeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def unfreeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad = True

    def unfreeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = True

    def freeze_last_head(self):
        for param in self.heads[-1].parameters():
            param.requires_grad = False

    def unfreeze_last_head(self):
        for param in self.heads[-1].parameters():
            param.requires_grad = True

    def _initialize_head_weights(self):
        if self.head_init_mode == "xavier":
            nn.init.xavier_uniform_(self.heads[-1].weight)
            nn.init.zeros_(self.heads[-1].bias)

        elif self.head_init_mode == "zeros":
            nn.init.zeros_(self.heads[-1].weight)
            nn.init.zeros_(self.heads[-1].bias)

        elif self.head_init_mode == "kaiming":
            nn.init.kaiming_uniform_(self.heads[-1].weight)
            nn.init.zeros_(self.heads[-1].bias)

    def _initialize_weights(self):
        """Initialize weights using different strategies"""
        # TODO: add different initialization strategies
        pass
