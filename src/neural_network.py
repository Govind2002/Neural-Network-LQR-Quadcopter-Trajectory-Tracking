"""
neural_network.py  –  Two-layer MLNN with stable Lyapunov-based update law
Architecture:  f̂(x) = Ŵᵀ σ(V̂ᵀ xₐ)
Update law:    σ-modification (prevents weight drift / explosion)

Author: Muhammed Aslam A (2025EEA2819), IIT Delhi
"""
import numpy as np


def sigmoid(z):
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-np.clip(z, -500, 500))),
                    np.exp(np.clip(z,-500,500)) / (1.0 + np.exp(np.clip(z,-500,500))))


def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1.0 - s)


class MLNN:
    def __init__(self, n_input=12, n_hidden=20, n_output=4,
                 gamma_W=1.0, gamma_V=1.0,
                 k_w=0.01, k_v=0.01,
                 w_max=5.0,          # weight magnitude clamp
                 seed=42):
        rng = np.random.default_rng(seed)
        self.n_input  = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.gamma_W  = gamma_W
        self.gamma_V  = gamma_V
        self.k_w      = k_w
        self.k_v      = k_v
        self.w_max    = w_max

        lim_V = np.sqrt(6.0 / (n_input + 1 + n_hidden))
        lim_W = np.sqrt(6.0 / (n_hidden + n_output))
        self.V_hat = rng.uniform(-lim_V, lim_V, (n_input + 1, n_hidden))
        self.W_hat = rng.uniform(-lim_W, lim_W, (n_hidden, n_output))

    def augment(self, x):
        return np.concatenate([[1.0], x])

    def forward(self, x):
        xa    = self.augment(x)
        z     = self.V_hat.T @ xa
        sig   = sigmoid(z)
        f_hat = self.W_hat.T @ sig
        return f_hat, {'xa': xa, 'z': z, 'sig': sig}

    def update(self, e, cache, dt):
        """
        Lyapunov-based weight update with σ-modification.
        Weights are clamped after each update to prevent explosion.
        """
        xa  = cache['xa']
        z   = cache['z']
        sig = cache['sig']
        sp  = sigmoid_prime(z)

        # Output weight update:  ΔŴ = -Γ_W [σ eᵀ + k_w Ŵ]
        dW = -self.gamma_W * (np.outer(sig, e) + self.k_w * self.W_hat)
        self.W_hat += dW * dt

        # Hidden weight update:  ΔV̂ = -Γ_V [xₐ (σ'⊙(Ŵ e))ᵀ + k_v V̂]
        We    = self.W_hat @ e
        sp_We = sp * We
        dV = -self.gamma_V * (np.outer(xa, sp_We) + self.k_v * self.V_hat)
        self.V_hat += dV * dt

        # Hard clamp weights to prevent unbounded growth
        np.clip(self.W_hat, -self.w_max, self.w_max, out=self.W_hat)
        np.clip(self.V_hat, -self.w_max, self.w_max, out=self.V_hat)
