import io
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import solve_ivp
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="Thermo-Viscous Recirculation PBE-PINN Workbench",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# 1. PROCESS & PHYSICS CONFIGURATION DATACLASS
# =====================================================================
@dataclass
class MillLoopConfig:
    # Geometric & Flow Parameters
    V_m: float = 0.50            # Active mill chamber volume (L)
    V_t: float = 4.50            # Recirculation holding tank volume (L)
    Q: float = 1.00              # Volumetric flow rate (L/min)
    t_max: float = 60.0          # Total batch grinding time (min)

    # Discretization Grid
    n_bins: int = 16             # Number of geometric size classes
    x_max: float = 25.0          # Coarsest size class x_0 (um)
    x_min: float = 0.10          # Finest collector/pan class x_{N-1} (um)

    # Slurry & Rheology Constants
    rho_slurry: float = 1150.0   # Slurry density (g/L)
    Cp_slurry: float = 3.85      # Specific heat capacity (J/(g*K))
    phi_v: float = 0.20          # Solids volume fraction (-)
    phi_max: float = 0.62        # Maximum packing fraction (-)
    T_ref: float = 298.15        # Reference temperature (K) [25.0 C]
    mu_ref: float = 1.8e-3       # Initial slurry viscosity at T_ref (Pa.s)
    E_mu_over_R: float = 1850.0  # Solvent Andrade/Arrhenius constant E_mu/R (K)

    # Thermal & Agitator Power Parameters
    P_agit_W: float = 650.0      # Nominal agitator shaft power at mu_ref (W)
    eta_diss: float = 0.88       # Fraction of shaft power dissipated as heat (-)
    alpha_P: float = 0.15        # Viscosity exponent on power draw (-)
    UA_mill: float = 18.0        # Mill jacket overall heat transfer coeff * area (W/K)
    UA_tank: float = 28.0        # Holding tank jacket UA (W/K)
    T_cool_m: float = 288.15     # Mill chiller coolant supply temp (K) [15.0 C]
    T_cool_t: float = 291.15     # Tank jacket coolant supply temp (K) [18.0 C]
    T_init: float = 298.15       # Initial homogenized loop temperature (K) [25.0 C]

    @property
    def tau_m(self) -> float:
        return self.V_m / self.Q

    @property
    def tau_t(self) -> float:
        return self.V_t / self.Q

    def get_size_grid(self) -> np.ndarray:
        """Returns geometric size grid ordered COARSEST (idx 0) to FINEST (idx N-1)."""
        return self.x_max * (self.x_min / self.x_max) ** (
            np.arange(self.n_bins) / (self.n_bins - 1)
        )

    def get_initial_psd(self, d50_feed: float = 9.5, sigma_g: float = 1.65) -> np.ndarray:
        """Generates a normalized log-normal initial feed mass fraction vector w_0."""
        x = self.get_size_grid()
        log_x = np.log(x)
        log_d50 = np.log(d50_feed)
        log_sig = np.log(sigma_g)
        w0 = np.exp(-0.5 * ((log_x - log_d50) / log_sig) ** 2)
        w0[-1] = 1e-5  # Negligible initial pan mass
        return w0 / np.sum(w0)


# =====================================================================
# 2. FORWARD DIGITAL TWIN (RK45 COUPLED PBE + THERMAL + RHEOLOGY SOLVER)
# =====================================================================
def assemble_numpy_base_kernel(
    x: np.ndarray, alpha: float, phi: float, gamma: float, beta: float
) -> np.ndarray:
    """Assembles the time-invariant dimensionless matrix K_tilde = (b - I) @ diag(s_hat)."""
    N = len(x)
    s_hat = (x / x[0]) ** alpha
    s_hat[-1] = 0.0  # Sink bin cannot break further

    x_upper = np.concatenate([[x[0]], x[:-1]])
    R = x_upper[:, None] / x[None, :]
    tril_mask = np.tril(np.ones((N, N)), k=-1)
    R_safe = np.clip(R * tril_mask, 1e-8, 1.0)

    B_cum = tril_mask * (phi * (R_safe ** gamma) + (1.0 - phi) * (R_safe ** beta))
    subdiag = np.eye(N, k=-1)
    B_cum = B_cum * (1.0 - subdiag) + subdiag

    B_shifted = np.vstack([B_cum[1:, :], np.zeros((1, N))])
    b_mat = (B_cum - B_shifted) * tril_mask
    K_tilde = (b_mat - np.eye(N)) * s_hat[None, :]
    return K_tilde


def compute_psd_quantiles_numpy(x: np.ndarray, w_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes D10, D50, D90, and Sauter Mean Diameter D32 across time.
    w_matrix: shape (T_steps, N) where index 0 is coarsest and N-1 is finest.
    """
    x_asc = x[::-1]
    log_x_asc = np.log(x_asc)
    w_asc = np.clip(w_matrix[:, ::-1], 0.0, 1.0)
    w_asc = w_asc / np.clip(np.sum(w_asc, axis=1, keepdims=True), 1e-12, None)
    q3_asc = np.cumsum(w_asc, axis=1)

    d10, d50, d90 = np.zeros(len(w_matrix)), np.zeros(len(w_matrix)), np.zeros(len(w_matrix))
    for k in range(len(w_matrix)):
        cdf_k = q3_asc[k]
        # Ensure strictly increasing grid for interpolation
        _, unique_idx = np.unique(cdf_k, return_index=True)
        if len(unique_idx) >= 2:
            d10[k] = np.exp(np.interp(0.10, cdf_k[unique_idx], log_x_asc[unique_idx]))
            d50[k] = np.exp(np.interp(0.50, cdf_k[unique_idx], log_x_asc[unique_idx]))
            d90[k] = np.exp(np.interp(0.90, cdf_k[unique_idx], log_x_asc[unique_idx]))
        else:
            d10[k] = d50[k] = d90[k] = x_asc[0]

    d32 = 1.0 / np.sum(w_matrix / x[None, :], axis=1)
    return d10, d50, d90, d32


def simulate_coupled_mill_tank_loop(
    cfg: MillLoopConfig,
    true_params: Dict[str, float],
    w0: np.ndarray,
    t_eval: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Integrates the coupled 2N + 2 non-isothermal PBE + CSTR recirculation loop via RK45.
    State vector y = [w_m (N), w_t (N), T_m (1), T_t (1)].
    """
    x = cfg.get_size_grid()
    N = cfg.n_bins
    d32_init = float(1.0 / np.sum(w0 / x))
    crowd_factor = (1.0 - cfg.phi_v / cfg.phi_max) ** (-2.5 * cfg.phi_max)

    S0_ref = true_params["S0_ref"]
    alpha = true_params["alpha"]
    phi = true_params["phi"]
    gamma = true_params["gamma"]
    beta = true_params["beta"]
    nu_mu = true_params["nu_mu"]
    kappa_A = true_params["kappa_A"]
    Ea_over_R = true_params.get("Ea_over_R", 0.0)

    K_tilde = assemble_numpy_base_kernel(x, alpha, phi, gamma, beta)

    # Convert W (J/s) to J/min for time in minutes
    sec_per_min = 60.0
    C_mill = cfg.rho_slurry * cfg.Cp_slurry * cfg.V_m  # J/K
    C_tank = cfg.rho_slurry * cfg.Cp_slurry * cfg.V_t  # J/K

    def rhs(t, y):
        w_m = np.clip(y[:N], 0.0, None)
        w_m = w_m / np.sum(w_m)
        w_t = np.clip(y[N:2*N], 0.0, None)
        w_t = w_t / np.sum(w_t)
        T_m = max(float(y[2*N]), 273.15)
        T_t = max(float(y[2*N + 1]), 273.15)

        # 1. Slurry effective viscosity mu_eff(T_m, w_m)
        inv_T_diff = (1.0 / T_m) - (1.0 / cfg.T_ref)
        mu_solvent = (cfg.mu_ref / crowd_factor) * np.exp(cfg.E_mu_over_R * inv_T_diff)
        d32_m = 1.0 / np.sum(w_m / x)
        area_ratio = max(d32_init / d32_m - 1.0, 0.0)
        mu_eff = mu_solvent * crowd_factor * np.exp(kappa_A * area_ratio)

        # 2. Thermo-viscous specific selection rate S0(t)
        S0_t = (
            S0_ref
            * np.exp(-Ea_over_R * inv_T_diff)
            * ((cfg.mu_ref / max(mu_eff, 1e-5)) ** nu_mu)
        )

        # 3. Coupled PBE ODEs
        dw_m_dt = (1.0 / cfg.tau_m) * (w_t - w_m) + S0_t * (K_tilde @ w_m)
        dw_t_dt = (1.0 / cfg.tau_t) * (w_m - w_t)

        # 4. Coupled Thermal Energy Balance ODEs (K/min)
        P_shaft_W = cfg.P_agit_W * ((mu_eff / cfg.mu_ref) ** cfg.alpha_P)
        Q_gen_J_min = sec_per_min * cfg.eta_diss * P_shaft_W
        Q_cool_m_J_min = sec_per_min * cfg.UA_mill * (T_m - cfg.T_cool_m)
        Q_cool_t_J_min = sec_per_min * cfg.UA_tank * (T_t - cfg.T_cool_t)

        dT_m_dt = (1.0 / cfg.tau_m) * (T_t - T_m) + (Q_gen_J_min - Q_cool_m_J_min) / C_mill
        dT_t_dt = (1.0 / cfg.tau_t) * (T_m - T_t) - Q_cool_t_J_min / C_tank

        return np.concatenate([dw_m_dt, dw_t_dt, [dT_m_dt, dT_t_dt]])

    y0 = np.concatenate([w0, w0, [cfg.T_init, cfg.T_init]])
    sol = solve_ivp(rhs, (0.0, cfg.t_max), y0, t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-8)

    w_m_traj = sol.y[:N, :].T
    w_t_traj = sol.y[N:2*N, :].T
    T_m_traj = sol.y[2*N, :]
    T_t_traj = sol.y[2*N + 1, :]

    # Recompute auxiliary trajectories (mu_eff, S0_t, quantiles)
    inv_T_diff = (1.0 / T_m_traj) - (1.0 / cfg.T_ref)
    mu_solv = (cfg.mu_ref / crowd_factor) * np.exp(cfg.E_mu_over_R * inv_T_diff)
    d10_m, d50_m, d90_m, d32_m = compute_psd_quantiles_numpy(x, w_m_traj)
    d10_t, d50_t, d90_t, d32_t = compute_psd_quantiles_numpy(x, w_t_traj)

    area_ratio = np.maximum(d32_init / d32_m - 1.0, 0.0)
    mu_eff_traj = mu_solv * crowd_factor * np.exp(kappa_A * area_ratio)
    S0_traj = S0_ref * np.exp(-Ea_over_R * inv_T_diff) * ((cfg.mu_ref / mu_eff_traj) ** nu_mu)

    return {
        "t": sol.t,
        "x": x,
        "w_m": w_m_traj,
        "w_t": w_t_traj,
        "T_m": T_m_traj,
        "T_t": T_t_traj,
        "mu_eff": mu_eff_traj,
        "S0_t": S0_traj,
        "d10_m": d10_m, "d50_m": d50_m, "d90_m": d90_m, "d32_m": d32_m,
        "d10_t": d10_t, "d50_t": d50_t, "d90_t": d90_t, "d32_t": d32_t,
        "d32_init": d32_init,
    }


# =====================================================================
# 3. PYTORCH THERMO-VISCOUS PBE KERNEL & DUAL-STATE PINN MODULES
# =====================================================================
class ThermoViscousPBEKernels(nn.Module):
    """
    Identifiability-constrained Selection (S) and Breakage (B) kernels with
    endogenous temperature & fines-surface-area viscosity coupling.
    """
    def __init__(
        self,
        cfg: MillLoopConfig,
        d32_init: float,
        init_params: Dict[str, float],
        learn_rheology: bool = True,
    ):
        super().__init__()
        x_grid = torch.tensor(cfg.get_size_grid(), dtype=torch.float32)
        self.register_buffer("x", x_grid)
        self.register_buffer("inv_x", 1.0 / x_grid)
        self.N = cfg.n_bins

        self.d32_init = d32_init
        self.T_ref = cfg.T_ref
        self.mu_ref = cfg.mu_ref
        self.E_mu_over_R = cfg.E_mu_over_R
        self.crowd_factor = float((1.0 - cfg.phi_v / cfg.phi_max) ** (-2.5 * cfg.phi_max))

        inv_sp = lambda y: torch.log(torch.expm1(torch.tensor(max(float(y), 1e-4), dtype=torch.float32)))
        inv_sig = lambda y: torch.logit(torch.tensor(np.clip(float(y), 1e-3, 1.0 - 1e-3), dtype=torch.float32))

        self.raw_S0_ref = nn.Parameter(inv_sp(init_params["S0_ref"]))
        self.raw_alpha = nn.Parameter(inv_sp(init_params["alpha"]))
        self.raw_phi = nn.Parameter(inv_sig(init_params["phi"]))
        self.raw_gamma = nn.Parameter(inv_sp(max(init_params["gamma"] - 0.1, 0.05)))
        self.raw_delta_beta = nn.Parameter(
            inv_sp(max(init_params["beta"] - init_params["gamma"] - 0.3, 0.1))
        )

        self.raw_nu_mu = nn.Parameter(inv_sp(init_params["nu_mu"]), requires_grad=learn_rheology)
        self.raw_kappa_A = nn.Parameter(inv_sp(init_params["kappa_A"]), requires_grad=learn_rheology)

    def set_rheology_trainable(self, trainable: bool):
        self.raw_nu_mu.requires_grad_(trainable)
        self.raw_kappa_A.requires_grad_(trainable)

    def get_physical_params(self) -> Tuple[torch.Tensor, ...]:
        S0_ref = F.softplus(self.raw_S0_ref)
        alpha = F.softplus(self.raw_alpha)
        phi = torch.sigmoid(self.raw_phi)
        gamma = 0.10 + F.softplus(self.raw_gamma)
        beta = gamma + 0.30 + F.softplus(self.raw_delta_beta)
        nu_mu = F.softplus(self.raw_nu_mu)
        kappa_A = F.softplus(self.raw_kappa_A)
        return S0_ref, alpha, phi, gamma, beta, nu_mu, kappa_A

    def get_param_dict(self) -> Dict[str, float]:
        S0_ref, alpha, phi, gamma, beta, nu_mu, kappa_A = self.get_physical_params()
        return {
            "S0_ref": float(S0_ref.detach().cpu()),
            "alpha": float(alpha.detach().cpu()),
            "phi": float(phi.detach().cpu()),
            "gamma": float(gamma.detach().cpu()),
            "beta": float(beta.detach().cpu()),
            "nu_mu": float(nu_mu.detach().cpu()),
            "kappa_A": float(kappa_A.detach().cpu()),
        }

    def compute_slurry_viscosity(self, T_m: torch.Tensor, w_m: torch.Tensor) -> torch.Tensor:
        _, _, _, _, _, _, kappa_A = self.get_physical_params()
        inv_T_diff = (1.0 / T_m) - (1.0 / self.T_ref)
        mu_solvent = (self.mu_ref / self.crowd_factor) * torch.exp(self.E_mu_over_R * inv_T_diff)

        inv_d32 = torch.matmul(w_m, self.inv_x.unsqueeze(-1))  # (batch, 1)
        area_ratio = F.softplus(10.0 * (self.d32_init * inv_d32 - 1.0)) / 10.0
        mu_eff = mu_solvent * self.crowd_factor * torch.exp(kappa_A * area_ratio)
        return mu_eff

    def assemble_base_kernel_matrix(self) -> torch.Tensor:
        _, alpha, phi, gamma, beta, _, _ = self.get_physical_params()

        s_hat = (self.x / self.x[0]) ** alpha
        mask_S = torch.ones_like(s_hat)
        mask_S[-1] = 0.0
        s_hat = s_hat * mask_S

        x_upper = torch.cat([self.x[:1], self.x[:-1]], dim=0)
        R = x_upper.unsqueeze(1) / self.x.unsqueeze(0)
        tril_mask = torch.tril(torch.ones(self.N, self.N, device=self.x.device), diagonal=-1)

        R_safe = torch.clamp(R * tril_mask, min=1e-8, max=1.0)
        B_cum = tril_mask * (phi * (R_safe ** gamma) + (1.0 - phi) * (R_safe ** beta))

        subdiag_mask = torch.diag(torch.ones(self.N - 1, device=self.x.device), diagonal=-1)
        B_cum = B_cum * (1.0 - subdiag_mask) + subdiag_mask

        B_shifted = torch.cat([B_cum[1:, :], torch.zeros(1, self.N, device=self.x.device)], dim=0)
        b_mat = (B_cum - B_shifted) * tril_mask

        I_mat = torch.eye(self.N, device=self.x.device)
        K_tilde = (b_mat - I_mat) * s_hat.unsqueeze(0)
        return K_tilde

    def compute_mill_breakage_rate(
        self, w_m: torch.Tensor, T_m: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        S0_ref, _, _, _, _, nu_mu, _ = self.get_physical_params()
        K_tilde = self.assemble_base_kernel_matrix()
        mu_eff = self.compute_slurry_viscosity(T_m, w_m)

        viscous_factor = (self.mu_ref / torch.clamp(mu_eff, min=1e-5)) ** nu_mu
        S0_t = S0_ref * viscous_factor

        base_rate = torch.matmul(w_m, K_tilde.T)
        net_breakage_rate = S0_t * base_rate
        return net_breakage_rate, S0_t, mu_eff


class CoupledThermoViscousPINN(nn.Module):
    """
    Predicts [w_m(t), w_t(t), T_m(t), T_t(t)] simultaneously.
    Uses forward-mode JVP (Jacobian-Vector Product) to compute exact time derivatives
    across all 2N + 2 outputs in a single forward pass.
    """
    def __init__(self, n_bins: int, t_max: float, T_ref: float = 298.15, hidden_dim: int = 64):
        super().__init__()
        self.n_bins = n_bins
        self.t_max = t_max
        self.T_ref = T_ref
        self.T_scale = 25.0  # Characteristic temperature rise scale (K)

        self.trunk = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.head_w_m = nn.Linear(hidden_dim, n_bins)
        self.head_w_t = nn.Linear(hidden_dim, n_bins)
        self.head_T = nn.Linear(hidden_dim, 2)  # [delta_T_m, delta_T_t]

    def _raw_forward_concat(self, t: torch.Tensor) -> torch.Tensor:
        t_norm = t / self.t_max
        h = self.trunk(t_norm)
        w_m = F.softmax(self.head_w_m(h), dim=-1)
        w_t = F.softmax(self.head_w_t(h), dim=-1)
        T_both = self.T_ref + self.T_scale * self.head_T(h)
        return torch.cat([w_m, w_t, T_both], dim=-1)

    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self._raw_forward_concat(t)
        N = self.n_bins
        return out[:, :N], out[:, N:2*N], out[:, 2*N:2*N+1], out[:, 2*N+1:2*N+2]

    def forward_with_jvp_time_derivatives(self, t: torch.Tensor):
        """
        Computes all states and their exact first time derivatives d/dt in ONE single JVP pass.
        """
        tangent = torch.ones_like(t)
        out, dout_dt = torch.autograd.functional.jvp(
            self._raw_forward_concat, (t,), (tangent,), create_graph=True
        )
        N = self.n_bins
        w_m, w_t = out[:, :N], out[:, N:2*N]
        T_m, T_t = out[:, 2*N:2*N+1], out[:, 2*N+1:2*N+2]

        dw_m_dt, dw_t_dt = dout_dt[:, :N], dout_dt[:, N:2*N]
        dT_m_dt, dT_t_dt = dout_dt[:, 2*N:2*N+1], dout_dt[:, 2*N+1:2*N+2]
        return w_m, w_t, T_m, T_t, dw_m_dt, dw_t_dt, dT_m_dt, dT_t_dt


# =====================================================================
# 4. COMPOSITE PHYSICS + DATA LOSS & FISHER INFORMATION IDENTIFIABILITY
# =====================================================================
def evaluate_coupled_pinn_losses(
    pinn: CoupledThermoViscousPINN,
    kernels: ThermoViscousPBEKernels,
    cfg: MillLoopConfig,
    t_obs: torch.Tensor,
    w_t_obs: torch.Tensor,
    T_t_obs: torch.Tensor,
    mu_obs: Optional[torch.Tensor],
    t_colloc: torch.Tensor,
    w0_tensor: torch.Tensor,
):
    # 1. Holding Tank Observation Data Loss (PDF + Cumulative Q3 CDF + Tank Temperature)
    w_m_obs, w_t_pred, T_m_obs_pred, T_t_pred = pinn(t_obs)
    loss_pdf = F.mse_loss(w_t_pred, w_t_obs)

    q3_pred = torch.flip(torch.cumsum(torch.flip(w_t_pred, dims=[-1]), dim=-1), dims=[-1])
    q3_exp = torch.flip(torch.cumsum(torch.flip(w_t_obs, dims=[-1]), dim=-1), dims=[-1])
    loss_cdf = F.mse_loss(q3_pred, q3_exp)

    loss_T_data = F.mse_loss((T_t_pred - cfg.T_ref) / 15.0, (T_t_obs - cfg.T_ref) / 15.0)
    loss_data = loss_pdf + 0.5 * loss_cdf + 0.25 * loss_T_data

    if mu_obs is not None:
        mu_pred_obs = kernels.compute_slurry_viscosity(T_m_obs_pred, w_m_obs)
        loss_visc = F.mse_loss(torch.log(mu_pred_obs), torch.log(mu_obs))
        loss_data = loss_data + 0.15 * loss_visc

    # 2. Initial Condition Loss at t = 0 (Pre-homogenized recirculation loop)
    t_zero = torch.zeros((1, 1), dtype=torch.float32, device=t_obs.device)
    w_m0, w_t0, T_m0, T_t0, _, dw_t0_dt, _, _ = pinn.forward_with_jvp_time_derivatives(t_zero)
    T_init_th = torch.full_like(T_m0, cfg.T_init)
    loss_ic = (
        F.mse_loss(w_t0, w0_tensor)
        + F.mse_loss(w_m0, w0_tensor)
        + F.mse_loss(cfg.tau_t * dw_t0_dt, torch.zeros_like(w0_tensor))
        + 0.1 * F.mse_loss((T_m0 - T_init_th) / 10.0, torch.zeros_like(T_m0))
        + 0.1 * F.mse_loss((T_t0 - T_init_th) / 10.0, torch.zeros_like(T_t0))
    )

    # 3. Coupled Physics Residuals at Collocation Points
    w_m_c, w_t_c, T_m_c, T_t_c, dw_m_dt, dw_t_dt, dT_m_dt, dT_t_dt = (
        pinn.forward_with_jvp_time_derivatives(t_colloc)
    )

    # 3a. Holding Tank Convective Buffer Residual (Dimensionless)
    res_tank_psd = cfg.tau_t * dw_t_dt - (w_m_c - w_t_c)
    loss_tank_pbe = torch.mean(res_tank_psd ** 2)

    # 3b. Mill Chamber Comminution + Convection Residual (Dimensionless)
    breakage_rate, _, mu_eff_c = kernels.compute_mill_breakage_rate(w_m_c, T_m_c)
    res_mill_psd = cfg.tau_m * dw_m_dt - (w_t_c - w_m_c) - cfg.tau_m * breakage_rate
    loss_mill_pbe = torch.mean(res_mill_psd ** 2)

    # 3c. Mill & Jacketed Holding Tank Energy Balance Residuals (Scaled by 15 K)
    sec_per_min = 60.0
    C_mill = cfg.rho_slurry * cfg.Cp_slurry * cfg.V_m
    C_tank = cfg.rho_slurry * cfg.Cp_slurry * cfg.V_t

    P_shaft_W = cfg.P_agit_W * ((torch.clamp(mu_eff_c, min=1e-5) / cfg.mu_ref) ** cfg.alpha_P)
    Q_gen_J_min = sec_per_min * cfg.eta_diss * P_shaft_W
    Q_cool_m_J_min = sec_per_min * cfg.UA_mill * (T_m_c - cfg.T_cool_m)
    Q_cool_t_J_min = sec_per_min * cfg.UA_tank * (T_t_c - cfg.T_cool_t)

    res_T_mill = (
        cfg.tau_m * dT_m_dt
        - (T_t_c - T_m_c)
        - cfg.tau_m * (Q_gen_J_min - Q_cool_m_J_min) / C_mill
    ) / 15.0
    res_T_tank = (
        cfg.tau_t * dT_t_dt
        - (T_m_c - T_t_c)
        + cfg.tau_t * (Q_cool_t_J_min / C_tank)
    ) / 15.0

    loss_thermal = torch.mean(res_T_mill ** 2) + torch.mean(res_T_tank ** 2)

    return loss_data, loss_ic, loss_tank_pbe, loss_mill_pbe, loss_thermal


def compute_fisher_correlation_matrix(
    pinn: CoupledThermoViscousPINN,
    kernels: ThermoViscousPBEKernels,
    t_colloc: torch.Tensor,
) -> Tuple[np.ndarray, list]:
    """
    Computes the parameter sensitivity Jacobian J_{c, p} = d(BreakageRate)/d(theta_p)
    and returns the normalized Fisher parameter correlation matrix.
    """
    param_names = ["S0_ref", "alpha", "phi", "gamma", "beta", "nu_mu", "kappa_A"]
    raw_params = [
        kernels.raw_S0_ref,
        kernels.raw_alpha,
        kernels.raw_phi,
        kernels.raw_gamma,
        kernels.raw_delta_beta,
        kernels.raw_nu_mu,
        kernels.raw_kappa_A,
    ]
    kernels.set_rheology_trainable(True)

    with torch.enable_grad():
        w_m_c, _, T_m_c, _ = pinn(t_colloc)
        w_m_c = w_m_c.detach()
        T_m_c = T_m_c.detach()
        rate_c, _, _ = kernels.compute_mill_breakage_rate(w_m_c, T_m_c)
        flat_rate = rate_c.reshape(-1)

        grads = []
        # Subsample every 4th residual entry for rapid interactive Jacobian evaluation
        sampled_entries = flat_rate[::4]
        for entry in sampled_entries:
            g = torch.autograd.grad(entry, raw_params, retain_graph=True, allow_unused=True)
            g_vec = [float(gi.detach().cpu()) if gi is not None else 0.0 for gi in g]
            grads.append(g_vec)

    J = np.array(grads, dtype=np.float64)
    FIM = J.T @ J + 1e-9 * np.eye(len(param_names))
    cov = np.linalg.inv(FIM)
    std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = cov / np.outer(std, std)
    return np.clip(corr, -1.0, 1.0), param_names


# =====================================================================
# SIDEBAR: PROCESS & KERNEL CONFIGURATION
# =====================================================================
st.sidebar.title("⚙️ Mill–Tank Loop Config")

with st.sidebar.expander("1. Volumes, Flow & Residence Time", expanded=True):
    V_m_in = st.slider("Mill Active Volume V_m (L)", 0.20, 2.00, 0.50, 0.05)
    V_t_in = st.slider("Holding Tank Volume V_t (L)", 1.00, 15.00, 4.50, 0.25)
    Q_in = st.slider("Recirculation Rate Q (L/min)", 0.25, 4.00, 1.00, 0.05)
    t_max_in = st.slider("Batch Grinding Duration (min)", 20.0, 120.0, 60.0, 5.0)

with st.sidebar.expander("2. Thermal & Rheology Constants", expanded=False):
    P_agit_in = st.slider("Nominal Shaft Power P_0 (W)", 200.0, 1500.0, 650.0, 50.0)
    UA_mill_in = st.slider("Mill Jacket UA (W/K)", 5.0, 50.0, 18.0, 1.0)
    UA_tank_in = st.slider("Tank Jacket UA (W/K)", 5.0, 60.0, 28.0, 1.0)
    T_cool_C = st.slider("Chiller Supply Temp (°C)", 5.0, 25.0, 15.0, 1.0)
    phi_v_in = st.slider("Solids Volume Fraction φ_v", 0.05, 0.40, 0.20, 0.01)

with st.sidebar.expander("3. Ground-Truth PBE Kernel Parameters", expanded=True):
    S0_true = st.slider("True S0_ref (min⁻¹)", 0.15, 1.50, 0.65, 0.05)
    alpha_true = st.slider("True Selection Exponent α", 0.50, 2.00, 1.15, 0.05)
    phi_true = st.slider("True Bimodal Weight φ", 0.20, 0.80, 0.55, 0.05)
    gamma_true = st.slider("True Fine Progeny Modulus γ", 0.35, 1.40, 0.75, 0.05)
    beta_true = st.slider("True Coarse Progeny Modulus β", 1.50, 4.00, 2.50, 0.10)
    nu_mu_true = st.slider("True Viscous Damping Exponent ν_μ", 0.10, 1.00, 0.45, 0.05)
    kappa_A_true = st.slider("True Fines Thickening Factor κ_A", 0.05, 0.60, 0.22, 0.02)

cfg = MillLoopConfig(
    V_m=V_m_in,
    V_t=V_t_in,
    Q=Q_in,
    t_max=t_max_in,
    P_agit_W=P_agit_in,
    UA_mill=UA_mill_in,
    UA_tank=UA_tank_in,
    T_cool_m=273.15 + T_cool_C,
    T_cool_t=273.15 + T_cool_C + 2.0,
    phi_v=phi_v_in,
)

true_params_dict = {
    "S0_ref": S0_true,
    "alpha": alpha_true,
    "phi": phi_true,
    "gamma": gamma_true,
    "beta": max(beta_true, gamma_true + 0.4),
    "nu_mu": nu_mu_true,
    "kappa_A": kappa_A_true,
}

# Run Forward Digital Twin Baseline
w0_vec = cfg.get_initial_psd()
t_dense = np.linspace(0.0, cfg.t_max, 241)
sim_data = simulate_coupled_mill_tank_loop(cfg, true_params_dict, w0_vec, t_dense)

# =====================================================================
# MAIN HEADER & KPI SUMMARY BAR
# =====================================================================
st.title("⚙️ Coupled Thermo-Viscous Recirculation PBE-PINN Workbench")
st.markdown(
    "Deconvolve **holding-tank residence-time lag** ($\tau_t = V_t/Q$), **non-isothermal mill heating**, "
    "and **fines-driven slurry viscosity** to identify intrinsic population balance kernels $(\mathbf{S}, \mathbf{B})$ in PyTorch."
)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Mill Residence Time (τ_m)", f"{cfg.tau_m:.2f} min", f"V_m = {cfg.V_m:.2f} L")
kpi2.metric("Tank Lag Time (τ_t)", f"{cfg.tau_t:.2f} min", f"V_t/V_m = {cfg.V_t/cfg.V_m:.1f}x")
max_d90_lag = float(np.max(sim_data["d90_t"] - sim_data["d90_m"]))
kpi3.metric("Peak Tank-to-Mill D90 Lag", f"{max_d90_lag:.2f} µm", "Tank coarser than mill")
peak_dT = float(np.max(sim_data["T_m"] - sim_data["T_t"]))
kpi4.metric("Peak Mill-to-Tank ΔT", f"+{peak_dT:.2f} °C", f"Max T_m = {np.max(sim_data['T_m'])-273.15:.1f} °C")
kpi5.metric("Final Holding Tank D90", f"{sim_data['d90_t'][-1]:.2f} µm", f"D32 = {sim_data['d32_t'][-1]:.2f} µm")

# =====================================================================
# WORKBENCH TABS
# =====================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Forward Digital Twin & Lag Explorer",
    "2️⃣ Experimental Tank Sampler & CSV I/O",
    "3️⃣ 3-Stage Curriculum Inverse PINN",
    "4️⃣ Deconvolved Mill vs. Tank Diagnostics",
])

# ---------------------------------------------------------------------
# TAB 1: FORWARD DIGITAL TWIN & LAG EXPLORER
# ---------------------------------------------------------------------
with tab1:
    st.subheader("Coupled Mill Chamber vs. Holding Tank Trajectories")
    fig_fwd = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Particle Size Quantiles D90 & D50 (Tank vs. Latent Mill)",
            "Non-Isothermal Slurry Temperatures T_m(t) vs. T_t(t)",
            "Effective Slurry Viscosity μ_eff(t) & Specific Rate S_0(t)",
            "Full Particle Size Distribution Shift (Mill vs. Tank)",
        ),
        specs=[[{}, {}], [{"secondary_y": True}, {}]],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    # Subplot (1,1): Quantiles
    fig_fwd.add_trace(go.Scatter(x=sim_data["t"], y=sim_data["d90_t"], name="Tank D90 (Sampled)", line=dict(color="#1f77b4", width=3)), row=1, col=1)
    fig_fwd.add_trace(go.Scatter(x=sim_data["t"], y=sim_data["d90_m"], name="Mill D90 (Latent)", line=dict(color="#ff7f0e", width=2.5, dash="dash")), row=1, col=1)
    fig_fwd.add_trace(go.Scatter(x=sim_data["t"], y=sim_data["d50_t"], name="Tank D50", line=dict(color="#2ca02c", width=2)), row=1, col=1)
    fig_fwd.add_trace(go.Scatter(x=sim_data["t"], y=sim_data["d50_m"], name="Mill D50", line=dict(color="#d62728", width=2, dash="dot")), row=1, col=1)

    # Subplot (1,2): Temperatures
    fig_fwd.add_trace(go.Scatter(x=sim_data["t"], y=sim_data["T_m"] - 273.15, name="Mill Chamber T_m (°C)", line=dict(color="#d62728", width=3)), row=1, col=2)
    fig_fwd.add_trace(go.Scatter(x=sim_data["t"], y=sim_data["T_t"] - 273.15, name="Holding Tank T_t (°C)", line=dict(color="#1f77b4", width=3, dash="dash")), row=1, col=2)

    # Subplot (2,1): Viscosity & S0(t)
    fig_fwd.add_trace(go.Scatter(x=sim_data["t"], y=sim_data["mu_eff"] * 1e3, name="Slurry Viscosity μ_eff (mPa·s)", line=dict(color="#9467bd", width=3)), row=2, col=1, secondary_y=False)
    fig_fwd.add_trace(go.Scatter(x=sim_data["t"], y=sim_data["S0_t"], name="Dynamic Selection Rate S0(t) (min⁻¹)", line=dict(color="#2ca02c", width=2.5, dash="dot")), row=2, col=1, secondary_y=True)

    # Subplot (2,2): PSD Snapshot Selector
    snap_min = st.slider("Inspect PSD Snapshot Timestamp (min)", 0.0, float(cfg.t_max), float(min(10.0, cfg.t_max / 3)), 1.0)
    idx_snap = int(np.argmin(np.abs(sim_data["t"] - snap_min)))
    fig_fwd.add_trace(go.Scatter(x=sim_data["x"], y=sim_data["w_m"][0], name="Initial Feed PSD (t=0)", line=dict(color="gray", dash="dot")), row=2, col=2)
    fig_fwd.add_trace(go.Scatter(x=sim_data["x"], y=sim_data["w_t"][idx_snap], name=f"Holding Tank PSD (t={snap_min:.0f}m)", line=dict(color="#1f77b4", width=3)), row=2, col=2)
    fig_fwd.add_trace(go.Scatter(x=sim_data["x"], y=sim_data["w_m"][idx_snap], name=f"Mill Chamber PSD (t={snap_min:.0f}m)", line=dict(color="#ff7f0e", width=3, dash="dash")), row=2, col=2)

    fig_fwd.update_xaxes(title_text="Time (min)", row=1, col=1)
    fig_fwd.update_yaxes(title_text="Particle Size (µm)", row=1, col=1)
    fig_fwd.update_xaxes(title_text="Time (min)", row=1, col=2)
    fig_fwd.update_yaxes(title_text="Temperature (°C)", row=1, col=2)
    fig_fwd.update_xaxes(title_text="Time (min)", row=2, col=1)
    fig_fwd.update_yaxes(title_text="Viscosity (mPa·s)", row=2, col=1, secondary_y=False)
    fig_fwd.update_yaxes(title_text="S0(t) (min⁻¹)", row=2, col=1, secondary_y=True)
    fig_fwd.update_xaxes(title_text="Particle Size Class x_i (µm)", type="log", row=2, col=2)
    fig_fwd.update_yaxes(title_text="Mass Fraction w_i", row=2, col=2)
    fig_fwd.update_layout(height=680, template="plotly_white", legend=dict(orientation="h", y=-0.12))
    st.plotly_chart(fig_fwd, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 2: EXPERIMENTAL TANK SAMPLER & CSV UPLOAD/DOWNLOAD
# ---------------------------------------------------------------------
with tab2:
    st.subheader("Configure Holding-Tank Sampling Schedule or Upload Lab CSV")
    col_s1, col_s2, col_s3 = st.columns(3)
    n_samples = col_s1.slider("Number of Offline Tank Samples (N_obs)", 6, 30, 14, 1)
    psd_noise_pct = col_s2.slider("Laser Diffraction Relative Noise (%)", 0.0, 8.0, 2.5, 0.5)
    temp_noise_C = col_s3.slider("Thermocouple Noise Std Dev (°C)", 0.0, 1.0, 0.15, 0.05)

    # Generate synthetic noisy observations from holding tank
    rng = np.random.default_rng(seed=42)
    sample_indices = np.linspace(0, len(sim_data["t"]) - 1, n_samples, dtype=int)
    t_obs_np = sim_data["t"][sample_indices]
    w_t_clean = sim_data["w_t"][sample_indices]

    noise_mult = 1.0 + (psd_noise_pct / 100.0) * rng.standard_normal(w_t_clean.shape)
    noise_add = 0.002 * (psd_noise_pct / 100.0) * np.abs(rng.standard_normal(w_t_clean.shape))
    w_t_noisy = np.clip(w_t_clean * noise_mult + noise_add, 1e-6, None)
    w_t_noisy = w_t_noisy / np.sum(w_t_noisy, axis=1, keepdims=True)

    T_t_noisy = sim_data["T_t"][sample_indices] + temp_noise_C * rng.standard_normal(n_samples)
    mu_noisy = sim_data["mu_eff"][sample_indices] * (1.0 + 0.02 * rng.standard_normal(n_samples))

    # Build DataFrame for inspection & CSV export
    bin_cols = [f"w_t_bin_{i}_{sim_data['x'][i]:.2f}um" for i in range(cfg.n_bins)]
    df_obs = pd.DataFrame(w_t_noisy, columns=bin_cols)
    df_obs.insert(0, "mu_eff_Pas", mu_noisy)
    df_obs.insert(0, "T_tank_K", T_t_noisy)
    df_obs.insert(0, "time_min", t_obs_np)

    uploaded_csv = st.file_uploader("Upload Custom Holding Tank PSD + Temperature CSV (Optional)", type=["csv"])
    if uploaded_csv is not None:
        df_obs = pd.read_csv(uploaded_csv)
        st.success(f"Loaded {len(df_obs)} experimental rows from uploaded CSV.")

    st.dataframe(df_obs.style.format("{:.4f}"), use_container_width=True, height=280)

    csv_buf = io.StringIO()
    df_obs.to_csv(csv_buf, index=False)
    st.download_button(
        "📥 Download Holding-Tank Experimental Template CSV",
        data=csv_buf.getvalue(),
        file_name="holding_tank_psd_thermo_samples.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------
# TAB 3: 3-STAGE CURRICULUM INVERSE PINN ENGINE
# ---------------------------------------------------------------------
with tab3:
    st.subheader("3-Stage Curriculum Inverse PINN Parameter Identification")
    st.markdown(
        "**Stage 1:** Kinematic tank filtering & mill deconvolution ($\theta$ only) $\longrightarrow$ "
        "**Stage 2:** Algebraic kernel pre-conditioning ($\boldsymbol{\phi}$ only) $\longrightarrow$ "
        "**Stage 3:** Joint thermo-viscous PBE refinement ($\theta + \boldsymbol{\phi}$)."
    )

    c_p1, c_p2, c_p3, c_p4 = st.columns(4)
    s1_epochs = c_p1.number_input("Stage 1 Epochs (Deconvolution)", 100, 2000, 450, 50)
    s2_epochs = c_p2.number_input("Stage 2 Epochs (Kernel Pre-Fit)", 50, 1500, 250, 50)
    s3_epochs = c_p3.number_input("Stage 3 Epochs (Joint PINN)", 100, 3000, 600, 50)
    use_visc_sensor = c_p4.checkbox("Include Inline Viscometer Loss L_μ", value=True)

    run_pinn_btn = st.button("🚀 Run 3-Stage Inverse PINN Training", type="primary", use_container_width=True)

    if run_pinn_btn:
        t_obs_th = torch.tensor(df_obs["time_min"].values[:, None], dtype=torch.float32)
        T_t_obs_th = torch.tensor(df_obs["T_tank_K"].values[:, None], dtype=torch.float32)
        mu_obs_th = (
            torch.tensor(df_obs["mu_eff_Pas"].values[:, None], dtype=torch.float32)
            if use_visc_sensor
            else None
        )
        w_cols = [c for c in df_obs.columns if c.startswith("w_t_bin_")]
        w_t_obs_th = torch.tensor(df_obs[w_cols].values, dtype=torch.float32)
        w0_th = torch.tensor(w0_vec[None, :], dtype=torch.float32)
        t_colloc_th = torch.linspace(0.0, cfg.t_max, 120).unsqueeze(-1)

        # Initialize deliberately perturbed initial kernel guess (30-40% off ground truth)
        init_guess = {
            "S0_ref": 0.42,
            "alpha": 0.85,
            "phi": 0.40,
            "gamma": 1.05,
            "beta": 1.95,
            "nu_mu": 0.25,
            "kappa_A": 0.12,
        }

        pinn_net = CoupledThermoViscousPINN(cfg.n_bins, cfg.t_max, cfg.T_ref, hidden_dim=64)
        kernel_net = ThermoViscousPBEKernels(cfg, sim_data["d32_init"], init_guess, learn_rheology=False)

        loss_history = []
        param_history = []
        total_epochs = int(s1_epochs + s2_epochs + s3_epochs)
        prog_bar = st.progress(0, text="Initializing Stage 1: Kinematic Tank Deconvolution...")
        t_start = time.time()

        # STAGE 1: Train PINN Trunk/Heads on Data + Tank Convective Balance + Thermal Balance
        opt1 = torch.optim.Adam(pinn_net.parameters(), lr=3e-3)
        for ep in range(int(s1_epochs)):
            opt1.zero_grad()
            l_data, l_ic, l_tank, l_mill, l_therm = evaluate_coupled_pinn_losses(
                pinn_net, kernel_net, cfg, t_obs_th, w_t_obs_th, T_t_obs_th, mu_obs_th, t_colloc_th, w0_th
            )
            loss = l_data + 2.0 * l_ic + 1.5 * l_tank + 0.5 * l_therm
            loss.backward()
            opt1.step()
            if ep % 25 == 0:
                loss_history.append({"epoch": ep, "stage": "Stage 1", "L_data": float(l_data), "L_tank": float(l_tank), "L_mill": float(l_mill), "L_therm": float(l_therm)})
                param_history.append({"epoch": ep, **kernel_net.get_param_dict()})
                prog_bar.progress((ep + 1) / total_epochs, text=f"Stage 1 (Kinematic Deconvolution) — Epoch {ep+1}/{s1_epochs}")

        # STAGE 2: Freeze Network, Pre-Condition PBE Kernels on Deconvolved Mill State
        opt2 = torch.optim.Adam([p for p in kernel_net.parameters() if p.requires_grad], lr=1.5e-2)
        for ep in range(int(s2_epochs)):
            opt2.zero_grad()
            l_data, l_ic, l_tank, l_mill, l_therm = evaluate_coupled_pinn_losses(
                pinn_net, kernel_net, cfg, t_obs_th, w_t_obs_th, T_t_obs_th, mu_obs_th, t_colloc_th, w0_th
            )
            loss_s2 = l_mill + 0.2 * l_data
            loss_s2.backward()
            opt2.step()
            g_ep = int(s1_epochs) + ep
            if ep % 20 == 0:
                loss_history.append({"epoch": g_ep, "stage": "Stage 2", "L_data": float(l_data), "L_tank": float(l_tank), "L_mill": float(l_mill), "L_therm": float(l_therm)})
                param_history.append({"epoch": g_ep, **kernel_net.get_param_dict()})
                prog_bar.progress((g_ep + 1) / total_epochs, text=f"Stage 2 (Algebraic Kernel Pre-Fit) — Epoch {ep+1}/{s2_epochs}")

        # STAGE 3: Unfreeze Rheology Exponents & Jointly Refine Network + Kernels
        kernel_net.set_rheology_trainable(True)
        opt3 = torch.optim.Adam([
            {"params": pinn_net.parameters(), "lr": 8e-4},
            {"params": kernel_net.parameters(), "lr": 4e-3},
        ])
        for ep in range(int(s3_epochs)):
            opt3.zero_grad()
            l_data, l_ic, l_tank, l_mill, l_therm = evaluate_coupled_pinn_losses(
                pinn_net, kernel_net, cfg, t_obs_th, w_t_obs_th, T_t_obs_th, mu_obs_th, t_colloc_th, w0_th
            )
            loss_s3 = l_data + 2.0 * l_ic + 1.5 * l_tank + 1.5 * l_mill + 0.5 * l_therm
            loss_s3.backward()
            opt3.step()
            g_ep = int(s1_epochs + s2_epochs) + ep
            if ep % 25 == 0 or ep == int(s3_epochs) - 1:
                loss_history.append({"epoch": g_ep, "stage": "Stage 3", "L_data": float(l_data), "L_tank": float(l_tank), "L_mill": float(l_mill), "L_therm": float(l_therm)})
                param_history.append({"epoch": g_ep, **kernel_net.get_param_dict()})
                prog_bar.progress((g_ep + 1) / total_epochs, text=f"Stage 3 (Joint Physics Refinement) — Epoch {ep+1}/{s3_epochs}")

        elapsed = time.time() - t_start
        prog_bar.progress(1.0, text=f"3-Stage Curriculum Completed in {elapsed:.1f} s!")

        # Evaluate dense PINN predictions and Fisher Information Correlation Matrix
        with torch.no_grad():
            t_dense_th = torch.tensor(sim_data["t"][:, None], dtype=torch.float32)
            w_m_pinn, w_t_pinn, T_m_pinn, T_t_pinn = pinn_net(t_dense_th)
            _, S0_pinn, mu_pinn = kernel_net.compute_mill_breakage_rate(w_m_pinn, T_m_pinn)

        fim_corr, fim_names = compute_fisher_correlation_matrix(pinn_net, kernel_net, t_colloc_th)

        st.session_state["pinn_results"] = {
            "loss_df": pd.DataFrame(loss_history),
            "param_df": pd.DataFrame(param_history),
            "init_guess": init_guess,
            "learned_params": kernel_net.get_param_dict(),
            "w_m_pinn": w_m_pinn.numpy(),
            "w_t_pinn": w_t_pinn.numpy(),
            "T_m_pinn": T_m_pinn.squeeze(-1).numpy(),
            "T_t_pinn": T_t_pinn.squeeze(-1).numpy(),
            "mu_pinn": mu_pinn.squeeze(-1).numpy(),
            "S0_pinn": S0_pinn.squeeze(-1).numpy(),
            "fim_corr": fim_corr,
            "fim_names": fim_names,
        }

    if "pinn_results" in st.session_state:
        res = st.session_state["pinn_results"]
        # Parameter Recovery Comparison Table
        rows_comp = []
        for k in ["S0_ref", "alpha", "phi", "gamma", "beta", "nu_mu", "kappa_A"]:
            t_val = true_params_dict[k]
            i_val = res["init_guess"][k]
            p_val = res["learned_params"][k]
            err_pct = 100.0 * abs(p_val - t_val) / max(abs(t_val), 1e-6)
            rows_comp.append({
                "Kernel Parameter": k,
                "Initial Guess": i_val,
                "Ground Truth": t_val,
                "PINN Identified": p_val,
                "Relative Error (%)": err_pct,
            })
        st.markdown("#### Identified PBE & Thermo-Viscous Kernel Parameters")
        st.dataframe(
            pd.DataFrame(rows_comp).style.format({
                "Initial Guess": "{:.3f}",
                "Ground Truth": "{:.3f}",
                "PINN Identified": "{:.3f}",
                "Relative Error (%)": "{:.2f}%",
            }),
            use_container_width=True,
        )

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            df_l = res["loss_df"]
            fig_loss = go.Figure()
            for col_name, label in [("L_data", "Data Loss (Tank)"), ("L_tank", "Tank Buffer Residual"), ("L_mill", "Mill PBE Residual"), ("L_therm", "Thermal Residual")]:
                fig_loss.add_trace(go.Scatter(x=df_l["epoch"], y=df_l[col_name], name=label, mode="lines"))
            fig_loss.update_layout(title="3-Stage Curriculum Loss Convergence", xaxis_title="Epoch", yaxis_title="Loss (Log Scale)", yaxis_type="log", height=380, template="plotly_white")
            st.plotly_chart(fig_loss, use_container_width=True)

        with col_g2:
            fig_fim = go.Figure(data=go.Heatmap(
                z=res["fim_corr"],
                x=res["fim_names"],
                y=res["fim_names"],
                zmin=-1.0, zmax=1.0,
                colorscale="RdBu_r",
                text=np.round(res["fim_corr"], 2),
                texttemplate="%{text}",
            ))
            fig_fim.update_layout(title="Empirical Fisher Parameter Correlation Matrix", height=380, template="plotly_white")
            st.plotly_chart(fig_fim, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 4: DECONVOLVED MILL VS. TANK DIAGNOSTICS
# ---------------------------------------------------------------------
with tab4:
    st.subheader("PINN Deconvolved Latent Mill State vs. Measured Holding Tank State")
    if "pinn_results" not in st.session_state:
        st.info("Click **Run 3-Stage Inverse PINN Training** in Tab 3 to populate the deconvolved mill diagnostics.")
    else:
        res = st.session_state["pinn_results"]
        _, d50_m_p, d90_m_p, _ = compute_psd_quantiles_numpy(sim_data["x"], res["w_m_pinn"])
        _, d50_t_p, d90_t_p, _ = compute_psd_quantiles_numpy(sim_data["x"], res["w_t_pinn"])
        _, _, d90_obs, _ = compute_psd_quantiles_numpy(sim_data["x"], df_obs[[c for c in df_obs.columns if c.startswith("w_t_bin_")]].values)

        fig_diag = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                "Deconvolved Mill D90 vs. Noisy Holding-Tank Samples",
                "PINN Reconstructed Mill & Tank Temperatures (°C)",
            ),
        )
        # Left: D90 Deconvolution
        fig_diag.add_trace(go.Scatter(x=df_obs["time_min"], y=d90_obs, mode="markers", name="Noisy Tank Samples D90", marker=dict(size=9, color="#1f77b4", symbol="circle-open", line=dict(width=2))), row=1, col=1)
        fig_diag.add_trace(go.Scatter(x=sim_data["t"], y=d90_t_p, name="PINN Tank D90_t(t)", line=dict(color="#1f77b4", width=3)), row=1, col=1)
        fig_diag.add_trace(go.Scatter(x=sim_data["t"], y=sim_data["d90_m"], name="True Latent Mill D90_m(t)", line=dict(color="black", width=2, dash="dot")), row=1, col=1)
        fig_diag.add_trace(go.Scatter(x=sim_data["t"], y=d90_m_p, name="PINN Deconvolved Mill D90_m(t)", line=dict(color="#ff7f0e", width=3, dash="dash")), row=1, col=1)

        # Right: Temperatures
        fig_diag.add_trace(go.Scatter(x=df_obs["time_min"], y=df_obs["T_tank_K"] - 273.15, mode="markers", name="Noisy Tank T_t Samples", marker=dict(size=8, color="#1f77b4")), row=1, col=2)
        fig_diag.add_trace(go.Scatter(x=sim_data["t"], y=res["T_t_pinn"] - 273.15, name="PINN Tank T_t(t)", line=dict(color="#1f77b4", width=2.5)), row=1, col=2)
        fig_diag.add_trace(go.Scatter(x=sim_data["t"], y=sim_data["T_m"] - 273.15, name="True Mill T_m(t)", line=dict(color="black", width=2, dash="dot")), row=1, col=2)
        fig_diag.add_trace(go.Scatter(x=sim_data["t"], y=res["T_m_pinn"] - 273.15, name="PINN Deconvolved Mill T_m(t)", line=dict(color="#d62728", width=3, dash="dash")), row=1, col=2)

        fig_diag.update_xaxes(title_text="Time (min)", row=1, col=1)
        fig_diag.update_yaxes(title_text="D90 Particle Size (µm)", row=1, col=1)
        fig_diag.update_xaxes(title_text="Time (min)", row=1, col=2)
        fig_diag.update_yaxes(title_text="Temperature (°C)", row=1, col=2)
        fig_diag.update_layout(height=450, template="plotly_white", legend=dict(orientation="h", y=-0.18))
        st.plotly_chart(fig_diag, use_container_width=True)
