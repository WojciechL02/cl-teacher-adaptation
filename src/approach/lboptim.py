from torch.optim import Optimizer


class LieBracketOptimizer(Optimizer):
    def __init__(self, params, names=None, lr=1e-3, h=1e-3):
        """
        Optimizer which includes the Lie Bracket term in the gradient update.

        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
            lr (float): Learning rate.
            h (float): Scaling factor for the Lie Bracket regularization term.
        """
        defaults = dict(lr=lr, h=h)
        super(LieBracketOptimizer, self).__init__(params, defaults)
        self.names = names

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

            # for i, (name, param) in enumerate(zip(self.names, group['params'])):
            for i, param in enumerate(group['params']):
                if param.grad is None:
                    continue

                grad = param.grad.data

                # if "bn" not in name:
                if i < lie_bracket_len:
                    lie_bracket_term = lie_bracket[i] if lie_bracket[i] is not None else 0.0
                else:
                    lie_bracket_term = 0.0

                grad = grad.add(lie_bracket_term, alpha=-(h / 2))

                # Update rule: θ = θ - lr * (∇L_tilde - h/2 * Lie Bracket)
                param.data -= lr * grad
