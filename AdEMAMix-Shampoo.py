import math
import torch
from torch.optim import Optimizer
import torch.distributed as dist
from typing import List, Optional, Tuple, Callable, Union

class AdEMAMixDistributedShampoo(Optimizer):
    """
    AdEMAMix optimizer with Distributed Shampoo preconditioning.

    Combines the AdEMAMix optimizer with Shampoo’s second-order preconditioning.
    Supports distributed training via torch.distributed.

    Args:
        params (iterable): Iterable of parameters to optimize.
        lr (float, optional): Learning rate (default: 1e-3).
        betas (Tuple[float, float, float], optional): Coefficients used for computing
            running averages of gradient, squared gradient, and slow EMA (default: (0.9, 0.999, 0.9999)).
        eps (float, optional): Term added to denominator to improve numerical stability (default: 1e-8).
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 0).
        alpha (float, optional): Alpha parameter for AdEMAMix (default: 5.0).
        T_alpha_beta3 (Optional[int], optional): Time constant for alpha and beta3 scheduling (default: None).
        shampoo_decay (float, optional): Decay rate for Shampoo preconditioners (default: 0.9).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float, float] = (0.9, 0.999, 0.9999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        alpha: float = 5.0,
        T_alpha_beta3: Optional[int] = None,
        shampoo_decay: float = 0.9,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if len(betas) != 3:
            raise ValueError(f"Invalid betas length: {len(betas)}, expected 3.")
        if not all(0.0 <= beta < 1.0 for beta in betas):
            raise ValueError(f"Invalid betas: {betas}. Each beta must be in [0, 1).")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= shampoo_decay < 1.0:
            raise ValueError(f"Invalid shampoo_decay value: {shampoo_decay}. Must be in [0, 1).")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            alpha=alpha,
            T_alpha_beta3=T_alpha_beta3,
            shampoo_decay=shampoo_decay,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            Optional[float]: The loss if closure is provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Iterate over parameter groups
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            exp_avg_slows = []
            preconditioners1 = []
            preconditioners2 = []
            state_steps = []

            # Collect parameters and their states
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('AdEMAMixDistributedShampoo does not support sparse gradients')
                if not p.requires_grad:
                    continue

                params_with_grad.append(p)
                grad = p.grad
                grads.append(grad)

                state = self.state[p]
                if not state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_slow'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Initialize Shampoo preconditioners as identity matrices or scalars
                    if p.dim() >= 2:
                        state['preconditioner1'] = torch.eye(p.size(0), device=p.device, dtype=p.dtype)
                        state['preconditioner2'] = torch.eye(p.size(1), device=p.device, dtype=p.dtype)
                    else:
                        state['preconditioner1'] = torch.tensor(1.0, device=p.device, dtype=p.dtype)
                        state['preconditioner2'] = torch.tensor(1.0, device=p.device, dtype=p.dtype)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                exp_avg_slows.append(state['exp_avg_slow'])
                preconditioners1.append(state['preconditioner1'])
                preconditioners2.append(state['preconditioner2'])
                state_steps.append(state['step'])
                state['step'] += 1

            if not params_with_grad:
                continue  # Skip if no parameters to update in this group

            betas = group['betas']
            beta1, beta2, beta3 = betas
            alpha = group['alpha']
            T_alpha_beta3 = group['T_alpha_beta3']
            lr = group['lr']
            weight_decay = group['weight_decay']
            eps = group['eps']
            shampoo_decay = group['shampoo_decay']

            # Update Shampoo preconditioners in a distributed manner
            self._update_preconditioners_distributed(
                preconditioners1, preconditioners2, grads, group, shampoo_decay, eps
            )

            # Update parameters using AdEMAMix with Shampoo preconditioning
            self._update_adamemix_distributed_shampoo(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                exp_avg_slows,
                preconditioners1,
                preconditioners2,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                beta3=beta3,
                alpha=alpha,
                T_alpha_beta3=T_alpha_beta3,
                lr=lr,
                weight_decay=weight_decay,
                eps=eps,
            )

        return loss

    def _update_preconditioners_distributed(
        self,
        preconditioners1: List[torch.Tensor],
        preconditioners2: List[torch.Tensor],
        grads: List[torch.Tensor],
        group: dict,
        shampoo_decay: float,
        eps: float
    ):
        """
        Updates Shampoo preconditioners and synchronizes them across distributed workers.

        Args:
            preconditioners1 (List[torch.Tensor]): List of first preconditioners for each parameter.
            preconditioners2 (List[torch.Tensor]): List of second preconditioners for each parameter.
            grads (List[torch.Tensor]): List of gradients for each parameter.
            group (dict): Parameter group options.
            shampoo_decay (float): Decay rate for Shampoo preconditioners.
            eps (float): Small epsilon for numerical stability.
        """
        for pc1, pc2, grad in zip(preconditioners1, preconditioners2, grads):
            if grad.dim() >= 2:
                A = grad @ grad.t()  # [in_features, in_features]
                B = grad.t() @ grad  # [out_features, out_features]
            else:
                A = (grad ** 2).sum()
                B = A.clone()  # For 1D gradients, B is same as A

            # Update preconditioners with exponential moving average
            pc1.mul_(shampoo_decay).add_(A, alpha=1 - shampoo_decay)
            pc2.mul_(shampoo_decay).add_(B, alpha=1 - shampoo_decay)

            # Synchronize preconditioners across workers
            if dist.is_initialized():
                dist.all_reduce(pc1, op=dist.ReduceOp.SUM)
                dist.all_reduce(pc2, op=dist.ReduceOp.SUM)
                world_size = dist.get_world_size()
                pc1.div_(world_size)
                pc2.div_(world_size)

    def _update_adamemix_distributed_shampoo(
        self,
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        exp_avg_sqs: List[torch.Tensor],
        exp_avg_slows: List[torch.Tensor],
        preconditioners1: List[torch.Tensor],
        preconditioners2: List[torch.Tensor],
        steps: List[int],
        beta1: float,
        beta2: float,
        beta3: float,
        alpha: float,
        T_alpha_beta3: Optional[int],
        lr: float,
        weight_decay: float,
        eps: float,
    ):
        """
        Performs the AdEMAMix update with Shampoo preconditioning.

        Args:
            params (List[torch.Tensor]): List of parameters to update.
            grads (List[torch.Tensor]): List of gradients for each parameter.
            exp_avgs (List[torch.Tensor]): List of first moment estimates.
            exp_avg_sqs (List[torch.Tensor]): List of second moment estimates.
            exp_avg_slows (List[torch.Tensor]): List of slow EMA estimates.
            preconditioners1 (List[torch.Tensor]): List of first preconditioners.
            preconditioners2 (List[torch.Tensor]): List of second preconditioners.
            steps (List[int]): List of step counts for each parameter.
            beta1 (float): Coefficient for first moment.
            beta2 (float): Coefficient for second moment.
            beta3 (float): Coefficient for slow EMA.
            alpha (float): Alpha parameter for AdEMAMix.
            T_alpha_beta3 (Optional[int]): Time constant for scheduling.
            lr (float): Learning rate.
            weight_decay (float): Weight decay coefficient.
            eps (float): Small epsilon for numerical stability.
        """
        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            exp_avg_slow = exp_avg_slows[i]
            pc1 = preconditioners1[i]
            pc2 = preconditioners2[i]
            step = steps[i]

            # Bias corrections
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            # Schedule alpha_t and beta3_t
            if T_alpha_beta3 is not None and T_alpha_beta3 > 0:
                alpha_t = min(step * alpha / T_alpha_beta3, alpha)
                # Avoid division by zero
                if T_alpha_beta3 != step:
                    log_beta1 = math.log(beta1)
                    log_beta3 = math.log(beta3)
                    denominator = (1 - step / T_alpha_beta3) * log_beta3 + (step / T_alpha_beta3) * log_beta1
                    if denominator != 0:
                        beta3_t = min(math.exp((log_beta1 * log_beta3) / denominator), beta3)
                    else:
                        beta3_t = beta3
                else:
                    beta3_t = beta3
            else:
                alpha_t = alpha
                beta3_t = beta3

            # Update biased first moment estimate
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            # Update biased second raw moment estimate
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            # Update slow EMA
            exp_avg_slow.mul_(beta3_t).add_(grad, alpha=1 - beta3_t)

            # Compute bias-corrected second moment
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            # Compute step size
            step_size = lr / (bias_correction1 if bias_correction1 > 0 else 0.01)

            # Apply weight decay
            if weight_decay != 0:
                param.mul_(1 - lr * weight_decay)

            # Compute Shampoo preconditioned gradient
            if grad.dim() >= 2:
                # Safe inversion with added epsilon to diagonal
                inv_pc1 = torch.inverse(pc1 + torch.eye(pc1.size(0), device=pc1.device, dtype=pc1.dtype) * eps).sqrt()
                inv_pc2 = torch.inverse(pc2 + torch.eye(pc2.size(1), device=pc2.device, dtype=pc2.dtype) * eps).sqrt()

                # Precondition the gradient
                preconditioned_grad = inv_pc1 @ grad @ inv_pc2
            else:
                # For 1D gradients, use scalar preconditioning
                preconditioned_grad = grad / (pc1.sqrt() + eps)

            # Combine AdEMAMix update with Shampoo preconditioning
            combined_grad = (exp_avg + alpha_t * exp_avg_slow + preconditioned_grad) / 3  # Weighted average

            # Update parameters
            param.addcdiv_(combined_grad, denom, value=-step_size)

            # Optional: Gradient Clipping (Uncomment if needed)
            # torch.nn.utils.clip_grad_norm_(param, max_norm=1.0)

    def __repr__(self):
        return (f"{self.__class__.__name__}(lr={self.defaults['lr']}, "
                f"betas={self.defaults['betas']}, eps={self.defaults['eps']}, "
                f"weight_decay={self.defaults['weight_decay']}, alpha={self.defaults['alpha']}, "
                f"T_alpha_beta3={self.defaults['T_alpha_beta3']})")
