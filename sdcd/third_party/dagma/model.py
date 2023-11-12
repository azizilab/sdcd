import copy

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm.auto import tqdm

import wandb


class DagmaNonlinear:
    def __init__(self, model: nn.Module, verbose=False, dtype=torch.double):
        self.vprint = print if verbose else lambda *a, **k: None
        self.model = model
        self.dtype = dtype

    def log_mse_loss(self, output, target):
        n, d = target.shape
        loss = 0.5 * d * torch.log(1 / n * torch.sum((output - target) ** 2))
        return loss

    def minimize(
        self,
        max_iter,
        lr,
        lambda1,
        lambda2,
        mu,
        s,
        lr_decay=False,
        checkpoint=1000,
        tol=1e-6,
        pbar=None,
        log_wandb=False,
    ):
        self.vprint(f"\nMinimize s={s} -- lr={lr}")
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=(0.99, 0.999),
            weight_decay=mu * lambda2,
        )
        if lr_decay is True:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        obj_prev = 1e16
        for i in range(max_iter):
            optimizer.zero_grad()
            h_val = self.model.h_func(s)
            if h_val.item() < 0:
                self.vprint(f"Found h negative {h_val.item()} at iter {i}")
                return False
            X_hat = self.model(self.X)
            score = self.log_mse_loss(X_hat, self.X)
            l1_reg = lambda1 * self.model.fc1_l1_reg()
            obj = mu * (score + l1_reg) + h_val
            obj.backward()
            optimizer.step()
            if lr_decay and (i + 1) % 1000 == 0:  # every 1000 iters reduce lr
                scheduler.step()
            if i % checkpoint == 0 or i == max_iter - 1:
                obj_new = obj.item()
                self.vprint(f"\nInner iteration {i}")
                self.vprint(f"\th(W(model)): {h_val.item()}")
                self.vprint(f"\tscore(model): {obj_new}")
                if log_wandb:
                    wandb.log(
                        {
                            "score": obj_new,
                            "h_val": h_val.item(),
                            "inner_iter": i,
                            "s": s,
                            "lr": lr,
                            "mu": mu,
                        }
                    )

                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    pbar.update(max_iter - i)
                    break
                obj_prev = obj_new
            pbar.update(1)
        return True

    def fit(
        self,
        X,
        lambda1=0.02,
        lambda2=0.005,
        T=4,
        mu_init=0.1,
        mu_factor=0.1,
        s=1.0,
        warm_iter=5e4,
        max_iter=8e4,
        lr=0.0002,
        w_threshold=0.3,
        checkpoint=1000,
        log_wandb=False,
    ):
        torch.set_default_dtype(self.dtype)
        if type(X) == torch.Tensor:
            self.X = X.type(self.dtype)
        elif type(X) == np.ndarray:
            self.X = torch.from_numpy(X).type(self.dtype)
        else:
            ValueError("X should be numpy array or torch Tensor.")

        mu = mu_init
        if type(s) == list:
            if len(s) < T:
                self.vprint(
                    f"Length of s is {len(s)}, using last value in s for iteration t >= {len(s)}"
                )
                s = s + (T - len(s)) * [s[-1]]
        elif type(s) in [int, float]:
            s = T * [s]
        else:
            ValueError("s should be a list, int, or float.")
        with tqdm(total=(T - 1) * warm_iter + max_iter) as pbar:
            for i in range(int(T)):
                self.vprint(f"\nDagma iter t={i + 1} -- mu: {mu}", 30 * "-")
                success, s_cur = False, s[i]
                inner_iter = int(max_iter) if i == T - 1 else int(warm_iter)
                model_copy = copy.deepcopy(self.model)
                lr_decay = False
                while success is False:
                    success = self.minimize(
                        inner_iter,
                        lr,
                        lambda1,
                        lambda2,
                        mu,
                        s_cur,
                        lr_decay,
                        checkpoint=checkpoint,
                        pbar=pbar,
                        log_wandb=log_wandb,
                    )
                    if success is False:
                        self.model.load_state_dict(model_copy.state_dict().copy())
                        lr *= 0.5
                        lr_decay = True
                        if lr < 1e-10:
                            break  # lr is too small
                        s_cur = 1
                mu *= mu_factor
        W_est = self.model.fc1_to_adj()
        W_est[np.abs(W_est) < w_threshold] = 0
        return W_est
