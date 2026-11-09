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
    page_title="NETZSCH DeltaVita Thermo-Viscous PBE-PINN & 3D Loop Digital Twin",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# 1. NETZSCH DELTAVITA EQUIPMENT, CHAMBER LINER & BEAD MEDIA CATALOGS
# =====================================================================
NETZSCH_DELTAVITA_MODELS = {
    "DV 200 (0.20 L Lab Interchangeable)": {
        "V_m": 0.20,              # Active chamber volume (L)
        "V_t_default": 1.50,      # Typical recirculation tank volume (L)
        "Q_default": 0.40,        # Nominal recirculation pump rate (L/min)
        "P_agit_W": 300.0,        # Nominal shaft power at v_t=10 m/s, Y-ZrO2 (W)
        "A_mill_m2": 0.022,       # Active cylindrical cooling jacket area (m^2)
        "A_tank_m2": 0.085,       # Holding tank wetted jacket area (m^2)
        "wall_mm": 4.5,           # Chamber liner wall thickness (mm)
        "D_rotor_mm": 62.0,       # Pin/disc agitator outer diameter (mm)
        "D_chamber_mm": 76.0,     # Inner grinding chamber bore diameter (mm)
        "L_chamber_mm": 95.0,     # Active grinding chamber length (mm)
        "n_discs": 3,             # Number of agitator pin/disc planes
        "rpm_nominal_10ms": 3080, # RPM corresponding to v_t = 10.0 m/s
    },
    "DV 300 (0.30 L R&D / Late-Discovery)": {
        "V_m": 0.30,
        "V_t_default": 2.50,
        "Q_default": 0.60,
        "P_agit_W": 450.0,
        "A_mill_m2": 0.029,
        "A_tank_m2": 0.120,
        "wall_mm": 5.0,
        "D_rotor_mm": 70.0,
        "D_chamber_mm": 86.0,
        "L_chamber_mm": 115.0,
        "n_discs": 4,
        "rpm_nominal_10ms": 2728,
    },
    "DV 600 (0.60 L Pilot / Clinical Supply)": {
        "V_m": 0.60,
        "V_t_default": 6.00,
        "Q_default": 1.20,
        "P_agit_W": 900.0,
        "A_mill_m2": 0.046,
        "A_tank_m2": 0.215,
        "wall_mm": 6.0,
        "D_rotor_mm": 88.0,
        "D_chamber_mm": 108.0,
        "L_chamber_mm": 145.0,
        "n_discs": 5,
        "rpm_nominal_10ms": 2170,
    },
    "DV 2000 (2.00 L GMP Production)": {
        "V_m": 2.00,
        "V_t_default": 20.00,
        "Q_default": 4.00,
        "P_agit_W": 3000.0,
        "A_mill_m2": 0.103,
        "A_tank_m2": 0.480,
        "wall_mm": 8.0,
        "D_rotor_mm": 128.0,
        "D_chamber_mm": 158.0,
        "L_chamber_mm": 215.0,
        "n_discs": 6,
        "rpm_nominal_10ms": 1492,
    },
}

CHAMBER_LINER_MATERIALS = {
    "Stainless Steel (AISI 316L Pharma Ra <= 0.4 um)": {
        "family": "Stainless Steel",
        "k_wall_WmK": 16.2,
        "wear_factor": 1.45,
        "metal_free": False,
        "max_tip_speed_ms": 14.0,
        "shell_color": "#7f8c8d",
    },
    "Ceramic - NETZSCH-Ceram C (Sintered SiC)": {
        "family": "Ceramic (SiC)",
        "k_wall_WmK": 110.0,
        "wear_factor": 0.65,
        "metal_free": True,
        "max_tip_speed_ms": 16.0,
        "shell_color": "#2c3e50",
    },
    "Ceramic - NETZSCH-Ceram N (Silicon Nitride Si3N4)": {
        "family": "Ceramic (Si3N4)",
        "k_wall_WmK": 28.0,
        "wear_factor": 0.75,
        "metal_free": True,
        "max_tip_speed_ms": 15.5,
        "shell_color": "#34495e",
    },
    "Ceramic - NETZSCH-Ceram Z (Zirconia ZrO2)": {
        "family": "Ceramic (ZrO2)",
        "k_wall_WmK": 2.2,
        "wear_factor": 0.85,
        "metal_free": True,
        "max_tip_speed_ms": 13.0,
        "shell_color": "#f39c12",
    },
}

BEAD_MEDIA_TYPES = {
    "Zirconium Oxide - VitaBeaDs NaNo (Y-ZrO2, 6050 kg/m3)": {
        "category": "Zirconium",
        "rho_kgm3": 6050.0,
        "hardness_HV": 1250.0,
        "base_wear_ppm_per_kWh_kg": 18.0,
        "contaminant_label": "ZrO2 / Y2O3",
    },
    "Zirconium Oxide - Ceria-Stabilized (Ce-ZrO2, 6200 kg/m3)": {
        "category": "Zirconium",
        "rho_kgm3": 6200.0,
        "hardness_HV": 1150.0,
        "base_wear_ppm_per_kWh_kg": 29.0,
        "contaminant_label": "ZrO2 / CeO2",
    },
    "Ceramic - Zirconia-Toughened Alumina (ZTA, 4150 kg/m3)": {
        "category": "Ceramic",
        "rho_kgm3": 4150.0,
        "hardness_HV": 1400.0,
        "base_wear_ppm_per_kWh_kg": 38.0,
        "contaminant_label": "Al2O3 / ZrO2",
    },
    "Ceramic - Sintered High-Purity Alumina (Al2O3, 3850 kg/m3)": {
        "category": "Ceramic",
        "rho_kgm3": 3850.0,
        "hardness_HV": 1550.0,
        "base_wear_ppm_per_kWh_kg": 55.0,
        "contaminant_label": "Al2O3",
    },
    "Ceramic - Silicon Nitride (Si3N4, 3250 kg/m3)": {
        "category": "Ceramic",
        "rho_kgm3": 3250.0,
        "hardness_HV": 1500.0,
        "base_wear_ppm_per_kWh_kg": 22.0,
        "contaminant_label": "Si3N4",
    },
}

DEFAULT_CONSTANTS_ROWS = [
    {"Symbol": "rho_slurry", "Parameter Name": "Slurry Density", "Value": 1150.0, "Unit": "g/L", "Physical Role": "Thermal capacitance & specific energy denominator"},
    {"Symbol": "Cp_slurry", "Parameter Name": "Slurry Specific Heat Capacity", "Value": 3.85, "Unit": "J/(g·K)", "Physical Role": "Sensible heat accumulation in mill & tank"},
    {"Symbol": "phi_max", "Parameter Name": "Maximum Random Packing Fraction", "Value": 0.62, "Unit": "-", "Physical Role": "Krieger-Dougherty crowding asymptote"},
    {"Symbol": "T_ref", "Parameter Name": "Reference Rheology Temperature", "Value": 298.15, "Unit": "K", "Physical Role": "Baseline temperature (25 °C) for viscosity & S0"},
    {"Symbol": "mu_ref", "Parameter Name": "Reference Initial Slurry Viscosity", "Value": 0.0018, "Unit": "Pa·s", "Physical Role": "Baseline slurry viscosity at t=0 and T_ref"},
    {"Symbol": "E_mu_over_R", "Parameter Name": "Solvent Arrhenius Viscosity Constant (E_μ/R)", "Value": 1850.0, "Unit": "K", "Physical Role": "Thermal thinning rate of continuous liquid vehicle"},
    {"Symbol": "eta_diss", "Parameter Name": "Viscous Heat Dissipation Fraction", "Value": 0.88, "Unit": "-", "Physical Role": "Fraction of agitator shaft work converted to heat"},
    {"Symbol": "alpha_P", "Parameter Name": "Viscosity Power-Draw Exponent", "Value": 0.15, "Unit": "-", "Physical Role": "Shaft power sensitivity to slurry thickening"},
    {"Symbol": "h_slurry", "Parameter Name": "Internal Slurry Film Heat Transfer Coeff", "Value": 1400.0, "Unit": "W/(m²·K)", "Physical Role": "Agitated slurry-to-liner convective resistance"},
    {"Symbol": "h_coolant", "Parameter Name": "Jacket Coolant Film Heat Transfer Coeff", "Value": 2200.0, "Unit": "W/(m²·K)", "Physical Role": "Spiral cooling jacket convective resistance"},
    {"Symbol": "U_tank", "Parameter Name": "Holding Tank Overall Jacket U-Value", "Value": 380.0, "Unit": "W/(m²·K)", "Physical Role": "Secondary heat removal in jacketed holding tank"},
    {"Symbol": "eps_pack", "Parameter Name": "Kwade Bead Packing Voidage Factor", "Value": 0.38, "Unit": "-", "Physical Role": "Inter-bead void volume in g(φ_GM) collision model"},
    {"Symbol": "sigma_se", "Parameter Name": "Kwade Stress-Energy Efficiency Width", "Value": 0.85, "Unit": "-", "Physical Role": "Log-normal transition breadth around SE_opt"},
]

if "custom_constants_df" not in st.session_state:
    st.session_state["custom_constants_df"] = pd.DataFrame(DEFAULT_CONSTANTS_ROWS)


def get_const_dict() -> Dict[str, float]:
    df = st.session_state["custom_constants_df"]
    return {row["Symbol"]: float(row["Value"]) for _, row in df.iterrows()}


def rpm_to_tip_speed(rpm: float, d_rotor_mm: float) -> float:
    return float((np.pi * (d_rotor_mm * 1e-3) * rpm) / 60.0)


def tip_speed_to_rpm(v_tip_ms: float, d_rotor_mm: float) -> float:
    return float((60.0 * v_tip_ms) / (np.pi * max(d_rotor_mm * 1e-3, 1e-6)))


def compute_netzsch_chamber_ua(
    model_key: str,
    chamber_key: str,
    h_slurry_Wm2K: float = 1400.0,
    h_coolant_Wm2K: float = 2200.0,
    U_tank_Wm2K: float = 380.0,
) -> Tuple[float, float, float]:
    m_spec = NETZSCH_DELTAVITA_MODELS[model_key]
    c_spec = CHAMBER_LINER_MATERIALS[chamber_key]

    delta_m = m_spec["wall_mm"] * 1e-3
    k_wall = c_spec["k_wall_WmK"]

    R_total = (1.0 / max(h_slurry_Wm2K, 1.0)) + (delta_m / max(k_wall, 0.01)) + (1.0 / max(h_coolant_Wm2K, 1.0))
    U_mill = 1.0 / R_total
    UA_mill = U_mill * m_spec["A_mill_m2"]
    UA_tank = U_tank_Wm2K * m_spec["A_tank_m2"]
    return float(UA_mill), float(UA_tank), float(U_mill)


# =====================================================================
# 2. PROCESS & PHYSICS CONFIGURATION DATACLASS
# =====================================================================
@dataclass
class MillLoopConfig:
    V_m: float = 0.30
    V_t: float = 2.50
    Q: float = 0.60
    t_max: float = 60.0

    n_bins: int = 16
    x_max: float = 25.0
    x_min: float = 0.10

    rho_slurry: float = 1150.0
    Cp_slurry: float = 3.85
    phi_v: float = 0.20
    phi_max: float = 0.62
    T_ref: float = 298.15
    mu_ref: float = 1.8e-3
    E_mu_over_R: float = 1850.0

    P_agit_W: float = 450.0
    eta_diss: float = 0.88
    alpha_P: float = 0.15
    UA_mill: float = 26.5
    UA_tank: float = 45.6
    T_cool_m: float = 288.15
    T_cool_t: float = 290.15
    T_init: float = 298.15

    @property
    def tau_m(self) -> float:
        return self.V_m / self.Q

    @property
    def tau_t(self) -> float:
        return self.V_t / self.Q

    def get_size_grid(self) -> np.ndarray:
        return self.x_max * (self.x_min / self.x_max) ** (
            np.arange(self.n_bins) / (self.n_bins - 1)
        )

    def get_initial_psd(self, d50_feed: float = 9.5, sigma_g: float = 1.65) -> np.ndarray:
        x = self.get_size_grid()
        log_x = np.log(x)
        log_d50 = np.log(d50_feed)
        log_sig = np.log(sigma_g)
        w0 = np.exp(-0.5 * ((log_x - log_d50) / log_sig) ** 2)
        w0[-1] = 1e-5
        return w0 / np.sum(w0)


# =====================================================================
# 3. 3D CAD DIGITAL TWIN OF COMPLETE RECIRCULATION LOOP & MILL CHAMBER
# =====================================================================
def _generate_3d_tube_mesh(
    path_xyz: np.ndarray, radius: float, n_circ: int = 14
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates a 3D cylindrical surface mesh along an arbitrary 3D polyline path_xyz (M, 3)."""
    M = path_xyz.shape[0]
    theta = np.linspace(0.0, 2.0 * np.pi, n_circ)
    X_tube = np.zeros((M, n_circ))
    Y_tube = np.zeros((M, n_circ))
    Z_tube = np.zeros((M, n_circ))

    for i in range(M):
        if i == 0:
            tang = path_xyz[1] - path_xyz[0]
        elif i == M - 1:
            tang = path_xyz[-1] - path_xyz[-2]
        else:
            tang = path_xyz[i + 1] - path_xyz[i - 1]
        tang_norm = np.linalg.norm(tang)
        t_vec = tang / max(tang_norm, 1e-9)

        # Construct orthonormal frame (n_vec, b_vec) perpendicular to t_vec
        ref = np.array([0.0, 0.0, 1.0]) if abs(t_vec[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        n_vec = np.cross(t_vec, ref)
        n_vec /= max(np.linalg.norm(n_vec), 1e-9)
        b_vec = np.cross(t_vec, n_vec)

        circle_pts = (
            path_xyz[i, None, :]
            + radius * np.cos(theta)[:, None] * n_vec[None, :]
            + radius * np.sin(theta)[:, None] * b_vec[None, :]
        )
        X_tube[i, :] = circle_pts[:, 0]
        Y_tube[i, :] = circle_pts[:, 1]
        Z_tube[i, :] = circle_pts[:, 2]

    return X_tube, Y_tube, Z_tube


def build_3d_recirculation_loop_digital_twin(
    model_key: str,
    chamber_key: str,
    bead_key: str,
    rpm: float,
    d_gm_um: float,
    phi_gm: float,
    Q_flow_Lmin: float,
    T_tank_C: float,
    T_mill_C: float,
    T_cool_C: float,
    d90_tank_um: float,
    d90_mill_um: float,
    SE_max_mJ: float,
    color_mode: str = "Local Stress Energy SE_GM (mJ)",
    view_scope: str = "Full Recirculation Loop (Tank + Pump + HX + Mill)",
) -> go.Figure:
    """
    Constructs an interactive 3D WebGL Digital Twin of either:
      (A) The complete closed-loop recirculation skid: Jacketed Holding Tank (with mixer),
          Peristaltic/Lobe Feed Pump, Sanitary Feed/Return Piping, In-Line Trim Heat Exchanger,
          and NETZSCH DeltaVita Grinding Chamber, OR
      (B) High-detail close-up of the NETZSCH DeltaVita Grinding Chamber alone.
    """
    m_spec = NETZSCH_DELTAVITA_MODELS[model_key]
    c_spec = CHAMBER_LINER_MATERIALS[chamber_key]

    R_ch = m_spec["D_chamber_mm"] * 0.5
    R_rot = m_spec["D_rotor_mm"] * 0.5
    R_shaft = R_rot * 0.34
    R_jacket = R_ch + m_spec["wall_mm"] + 6.0
    L = m_spec["L_chamber_mm"]
    n_discs = m_spec["n_discs"]

    fig = go.Figure()
    theta_grid = np.linspace(0, 2 * np.pi, 40)
    x_cyl = np.linspace(0, L, 22)
    Theta, X_c = np.meshgrid(theta_grid, x_cyl)

    # -----------------------------------------------------------------
    # PART 1: NETZSCH DELTAVITA HORIZONTAL GRINDING CHAMBER (x in [0, L])
    # -----------------------------------------------------------------
    # 1a. Outer Cooling Jacket Shell
    fig.add_trace(
        go.Surface(
            x=X_c,
            y=R_jacket * np.cos(Theta),
            z=R_jacket * np.sin(Theta),
            surfacecolor=np.full_like(X_c, T_cool_C),
            colorscale=[[0, "#00b4d8"], [1, "#0077b6"]],
            showscale=False,
            opacity=0.15,
            name=f"Mill Cooling Jacket ({T_cool_C:.1f} °C)",
            hoverinfo="name",
        )
    )

    # 1b. Grinding Chamber Liner Wall
    T_axial_field = T_tank_C + (T_mill_C - T_tank_C) * (X_c / L) ** 0.75
    fig.add_trace(
        go.Surface(
            x=X_c,
            y=R_ch * np.cos(Theta),
            z=R_ch * np.sin(Theta),
            surfacecolor=T_axial_field,
            colorscale="Inferno",
            showscale=False,
            opacity=0.28,
            name=f"Mill Liner: {c_spec['family']}",
            hovertemplate="Mill Axial x: %{x:.1f} mm<br>Liner Temp: %{surfacecolor:.1f} °C<extra>Mill Chamber</extra>",
        )
    )

    # 1c. Central Agitator Rotor Shaft
    fig.add_trace(
        go.Surface(
            x=X_c,
            y=R_shaft * np.cos(Theta),
            z=R_shaft * np.sin(Theta),
            colorscale=[[0, "#576574"], [1, "#8395a7"]],
            showscale=False,
            opacity=0.95,
            name=f"Mill Rotor Shaft ({rpm:.0f} RPM)",
            hoverinfo="name",
        )
    )

    # 1d. Agitator Pin-Discs
    disc_positions = np.linspace(0.16 * L, 0.84 * L, n_discs)
    r_disc_grid = np.linspace(R_shaft, R_rot, 7)
    Theta_d, R_d = np.meshgrid(theta_grid, r_disc_grid)
    for idx_d, x_d in enumerate(disc_positions):
        fig.add_trace(
            go.Surface(
                x=np.full_like(R_d, x_d),
                y=R_d * np.cos(Theta_d),
                z=R_d * np.sin(Theta_d),
                colorscale=[[0, "#c8d6e5"], [1, "#ff9f43"]],
                showscale=False,
                opacity=0.88,
                name="Pin-Disc Agitator Elements",
                showlegend=(idx_d == 0),
                hovertemplate=f"Disc #{idx_d+1} at x={x_d:.1f} mm<br>Tip Speed: {rpm_to_tip_speed(rpm, m_spec['D_rotor_mm']):.2f} m/s<extra></extra>",
            )
        )

    # 1e. Dynamic Cartridge Separation Screen at Mill Outlet
    x_scr = np.linspace(0.87 * L, 0.97 * L, 7)
    Theta_s, X_s = np.meshgrid(theta_grid, x_scr)
    R_scr = R_shaft * 1.45
    fig.add_trace(
        go.Surface(
            x=X_s,
            y=R_scr * np.cos(Theta_s),
            z=R_scr * np.sin(Theta_s),
            colorscale=[[0, "#1dd1a1"], [1, "#10ac84"]],
            showscale=False,
            opacity=0.90,
            name="Dynamic Cartridge Separator",
            hoverinfo="name",
        )
    )

    # 1f. 3D Grinding Bead Cloud Inside Active Mill Annulus
    rng_3d = np.random.default_rng(seed=108)
    n_beads_viz = int(260 + 200 * (phi_gm - 0.60) / 0.30)
    x_beads = rng_3d.uniform(0.04 * L, 0.94 * L, n_beads_viz)
    theta_beads = rng_3d.uniform(0.0, 2.0 * np.pi, n_beads_viz)
    r_beads = rng_3d.triangular(R_shaft * 1.12, R_rot, R_ch * 0.97, n_beads_viz)

    y_beads = r_beads * np.cos(theta_beads)
    z_beads = r_beads * np.sin(theta_beads)

    dist_to_disc = np.min(np.abs(x_beads[:, None] - disc_positions[None, :]), axis=1)
    axial_decay = np.exp(-dist_to_disc / (0.12 * L))
    radial_shear = np.exp(-((r_beads - R_rot) / (0.18 * R_ch)) ** 2)

    local_SE_mJ = SE_max_mJ * (0.15 + 0.85 * radial_shear * (0.4 + 0.6 * axial_decay))
    local_Temp_C = (
        T_tank_C
        + (T_mill_C - T_tank_C) * (x_beads / L) ** 0.75
        + 1.8 * radial_shear
    )

    if color_mode == "Local Stress Energy SE_GM (mJ)":
        c_vals = local_SE_mJ
        c_title = "SE_GM (mJ)"
        c_scale = "Plasma"
    else:
        c_vals = local_Temp_C
        c_title = "Temp (°C)"
        c_scale = "Turbo"

    marker_px = float(np.clip(4.0 + (d_gm_um - 150.0) / 150.0, 4.0, 10.0))

    fig.add_trace(
        go.Scatter3d(
            x=x_beads,
            y=y_beads,
            z=z_beads,
            mode="markers",
            name=f"Grinding Beads ({bead_key[:16]}...)",
            marker=dict(
                size=marker_px,
                color=c_vals,
                colorscale=c_scale,
                opacity=0.92,
                colorbar=dict(title=c_title, len=0.62, thickness=15, x=0.97),
                line=dict(width=0.3, color="#222"),
            ),
            customdata=np.stack([local_SE_mJ, local_Temp_C, r_beads], axis=-1),
            hovertemplate=(
                "<b>Grinding Bead</b><br>"
                "Axial x: %{x:.1f} mm | Radial r: %{customdata[2]:.1f} mm<br>"
                "Stress Energy: %{customdata[0]:.4f} mJ<br>"
                "Local Temp: %{customdata[1]:.1f} °C<extra></extra>"
            ),
        )
    )

    # -----------------------------------------------------------------
    # PART 2: FULL RECIRCULATION SKID (HOLDING TANK + PUMP + TRIM HX + PIPING)
    # -----------------------------------------------------------------
    if view_scope.startswith("Full Recirculation Loop"):
        # Position Holding Tank at (-x_tank, y_tank, z_tank) scaled proportionally to mill L
        x_tk = -1.35 * L
        y_tk = -1.65 * L
        R_tk = 0.58 * L
        R_tk_jacket = R_tk * 1.09
        H_tk = 1.65 * L
        z_tk_bot = -0.75 * L
        z_tk_top = z_tk_bot + H_tk
        z_slurry_lvl = z_tk_bot + 0.78 * H_tk

        z_tank_grid = np.linspace(z_tk_bot, z_tk_top, 20)
        Theta_tk, Z_tk = np.meshgrid(theta_grid, z_tank_grid)

        # 2a. Holding Tank Outer Cooling Jacket
        fig.add_trace(
            go.Surface(
                x=x_tk + R_tk_jacket * np.cos(Theta_tk),
                y=y_tk + R_tk_jacket * np.sin(Theta_tk),
                z=Z_tk,
                colorscale=[[0, "#48cae4"], [1, "#0096c7"]],
                showscale=False,
                opacity=0.16,
                name="Holding Tank Cooling Jacket",
                hoverinfo="name",
            )
        )

        # 2b. Holding Tank Slurry Volume (Wetted Inner Cylinder up to 78% level)
        z_liq_grid = np.linspace(z_tk_bot, z_slurry_lvl, 16)
        Theta_lq, Z_lq = np.meshgrid(theta_grid, z_liq_grid)
        fig.add_trace(
            go.Surface(
                x=x_tk + R_tk * np.cos(Theta_lq),
                y=y_tk + R_tk * np.sin(Theta_lq),
                z=Z_lq,
                surfacecolor=np.full_like(Z_lq, T_tank_C),
                colorscale=[[0, "#38b000"], [1, "#70e000"]],
                showscale=False,
                opacity=0.42,
                name=f"Holding Tank Slurry (T_t={T_tank_C:.1f}°C, D90={d90_tank_um:.2f}µm)",
                hovertemplate=(
                    f"<b>Recirculation Holding Tank</b><br>"
                    f"Bulk Slurry Temp T_t: {T_tank_C:.2f} °C<br>"
                    f"Sampled Tank D90_t: {d90_tank_um:.2f} µm<br>"
                    f"Mill Outlet D90_m: {d90_mill_um:.2f} µm<extra></extra>"
                ),
            )
        )

        # 2c. Holding Tank Top Free-Surface Cap
        r_cap = np.linspace(0.0, R_tk, 6)
        Theta_cp, R_cp = np.meshgrid(theta_grid, r_cap)
        fig.add_trace(
            go.Surface(
                x=x_tk + R_cp * np.cos(Theta_cp),
                y=y_tk + R_cp * np.sin(Theta_cp),
                z=np.full_like(R_cp, z_slurry_lvl),
                colorscale=[[0, "#70e000"], [1, "#38b000"]],
                showscale=False,
                opacity=0.55,
                name="Tank Liquid Level",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # 2d. Holding Tank Agitator Shaft & Marine Impeller Blades
        z_imp_shaft = np.linspace(z_tk_bot + 0.18 * H_tk, z_tk_top + 0.15 * H_tk, 10)
        Theta_is, Z_is = np.meshgrid(theta_grid, z_imp_shaft)
        R_imp_sh = 0.06 * R_tk
        fig.add_trace(
            go.Surface(
                x=x_tk + R_imp_sh * np.cos(Theta_is),
                y=y_tk + R_imp_sh * np.sin(Theta_is),
                z=Z_is,
                colorscale=[[0, "#2f3640"], [1, "#718093"]],
                showscale=False,
                opacity=0.95,
                name="Tank Mixer Impeller",
                hoverinfo="name",
            )
        )
        # Impeller disc near bottom of tank
        r_imp_blade = np.linspace(R_imp_sh, 0.48 * R_tk, 5)
        Theta_ib, R_ib = np.meshgrid(theta_grid, r_imp_blade)
        fig.add_trace(
            go.Surface(
                x=x_tk + R_ib * np.cos(Theta_ib),
                y=y_tk + R_ib * np.sin(Theta_ib),
                z=np.full_like(R_ib, z_tk_bot + 0.22 * H_tk) + 0.05 * R_tk * np.sin(3 * Theta_ib),
                colorscale=[[0, "#fbc531"], [1, "#e1b12c"]],
                showscale=False,
                opacity=0.92,
                name="Tank Pitched-Blade Impeller",
                showlegend=False,
                hoverinfo="name",
            )
        )

        # 2e. Peristaltic / Lobe Feed Pump Head (Centered at x_pump, y_pump, z_pump)
        x_pump = -0.45 * L
        y_pump = -0.82 * L
        z_pump = z_tk_bot + 0.05 * L
        R_pump = 0.22 * L
        y_p_cyl = np.linspace(y_pump - 0.12 * L, y_pump + 0.12 * L, 8)
        Theta_p, Y_p = np.meshgrid(theta_grid, y_p_cyl)
        fig.add_trace(
            go.Surface(
                x=x_pump + R_pump * np.cos(Theta_p),
                y=Y_p,
                z=z_pump + R_pump * np.sin(Theta_p),
                colorscale=[[0, "#e84118"], [1, "#c23616"]],
                showscale=False,
                opacity=0.92,
                name=f"Recirculation Pump (Q = {Q_flow_Lmin:.2f} L/min)",
                hovertemplate=f"<b>Peristaltic Feed Pump</b><br>Flow Rate Q: {Q_flow_Lmin:.2f} L/min<extra></extra>",
            )
        )

        # 2f. Feed Line Pipe: Tank Bottom -> Pump -> Mill Inlet (x = 0.04*L, y = 0, z = -R_ch)
        r_pipe = 0.055 * L
        feed_waypoints = np.array([
            [x_tk + R_tk * 0.65, y_tk + R_tk * 0.65, z_tk_bot + 0.06 * L],
            [x_pump - 0.15 * L, y_pump, z_pump],
            [x_pump + 0.15 * L, y_pump, z_pump],
            [-0.08 * L, -0.25 * L, -0.55 * R_ch],
            [0.04 * L, 0.0, -0.92 * R_ch],
        ])
        # Interpolate smooth spline-like polyline for feed pipe
        t_wp = np.linspace(0, 1, len(feed_waypoints))
        t_fine = np.linspace(0, 1, 36)
        feed_path = np.column_stack([np.interp(t_fine, t_wp, feed_waypoints[:, k]) for k in range(3)])
        Xf_p, Yf_p, Zf_p = _generate_3d_tube_mesh(feed_path, r_pipe)
        fig.add_trace(
            go.Surface(
                x=Xf_p, y=Yf_p, z=Zf_p,
                colorscale=[[0, "#00a8ff"], [1, "#0097e6"]],
                showscale=False,
                opacity=0.85,
                name=f"Feed Pipe: Tank ➔ Mill ({T_tank_C:.1f} °C)",
                hoverinfo="name",
            )
        )

        # 2g. Return Line Pipe + In-Line Trim Heat Exchanger: Mill Outlet (x = 0.95*L) -> Tank Top
        return_waypoints = np.array([
            [0.95 * L, 0.0, R_ch * 0.92],
            [0.95 * L, -0.45 * L, z_tk_top + 0.12 * L],
            [0.25 * L, -0.95 * L, z_tk_top + 0.12 * L],
            [-0.55 * L, -1.35 * L, z_tk_top + 0.12 * L],
            [x_tk, y_tk + R_tk * 0.55, z_tk_top + 0.08 * L],
            [x_tk, y_tk + R_tk * 0.55, z_slurry_lvl],
        ])
        t_wp_r = np.linspace(0, 1, len(return_waypoints))
        return_path = np.column_stack([np.interp(t_fine, t_wp_r, return_waypoints[:, k]) for k in range(3)])
        Xr_p, Yr_p, Zr_p = _generate_3d_tube_mesh(return_path, r_pipe)
        fig.add_trace(
            go.Surface(
                x=Xr_p, y=Yr_p, z=Zr_p,
                colorscale=[[0, "#ff6b6b"], [1, "#ee5253"]],
                showscale=False,
                opacity=0.88,
                name=f"Return Pipe: Mill ➔ Tank ({T_mill_C:.1f} °C)",
                hoverinfo="name",
            )
        )

        # 2h. In-Line Shell-and-Tube Trim Heat Exchanger on Return Line
        hx_seg = return_path[12:22]
        X_hx, Y_hx, Z_hx = _generate_3d_tube_mesh(hx_seg, r_pipe * 2.35)
        fig.add_trace(
            go.Surface(
                x=X_hx, y=Y_hx, z=Z_hx,
                colorscale=[[0, "#00d2d3"], [1, "#01a3a4"]],
                showscale=False,
                opacity=0.78,
                name="In-Line Trim Heat Exchanger (Return Line)",
                hoverinfo="name",
            )
        )

        # 2i. 3D Text Annotations Above Equipment Units
        fig.add_trace(
            go.Scatter3d(
                x=[0.5 * L, x_tk, x_pump, float(np.mean(hx_seg[:, 0]))],
                y=[0.0, y_tk, y_pump, float(np.mean(hx_seg[:, 1]))],
                z=[R_jacket * 1.45, z_tk_top + 0.28 * L, z_pump + R_pump * 1.5, float(np.mean(hx_seg[:, 2])) + 0.22 * L],
                mode="text",
                text=[
                    f"<b>NETZSCH {model_key[:6]} Mill</b><br>T_m={T_mill_C:.1f}°C | D90_m={d90_mill_um:.2f}µm",
                    f"<b>Jacketed Holding Tank (Sampling Port)</b><br>T_t={T_tank_C:.1f}°C | D90_t={d90_tank_um:.2f}µm",
                    f"<b>Feed Pump</b> ({Q_flow_Lmin:.2f} L/min)",
                    "<b>In-Line Trim HX</b>",
                ],
                textfont=dict(size=11, color="#1e272e"),
                name="Equipment Labels",
                showlegend=False,
                hoverinfo="skip",
            )
        )
        cam_eye = dict(x=1.25, y=-2.15, z=1.15)
        plot_height = 650
    else:
        cam_eye = dict(x=1.45, y=-1.65, z=0.95)
        plot_height = 560

    fig.update_layout(
        title=dict(
            text=(
                f"3D Digital Twin ({view_scope.split(' (')[0]}): {model_key[:6]} | "
                f"Liner: {c_spec['family']} | {rpm:.0f} RPM ({rpm_to_tip_speed(rpm, m_spec['D_rotor_mm']):.2f} m/s)"
            ),
            font=dict(size=15),
        ),
        scene=dict(
            xaxis_title="X Coordinate (mm)",
            yaxis_title="Y Coordinate (mm)",
            zaxis_title="Z Elevation (mm)",
            aspectmode="data",
            camera=dict(eye=cam_eye),
            bgcolor="#f8f9fa",
        ),
        height=plot_height,
        margin=dict(l=0, r=0, t=45, b=0),
        legend=dict(orientation="h", y=0.01, x=0.01, bgcolor="rgba(255,255,255,0.80)"),
    )
    return fig


# =====================================================================
# 4. FORWARD DIGITAL TWIN & KWADE STRESS-ENERGY SCALING FUNCTIONS
# =====================================================================
def assemble_numpy_base_kernel(
    x: np.ndarray, alpha: float, phi: float, gamma: float, beta: float
) -> np.ndarray:
    N = len(x)
    s_hat = (x / x[0]) ** alpha
    s_hat[-1] = 0.0

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


def compute_psd_quantiles_numpy(
    x: np.ndarray, w_matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_asc = x[::-1]
    log_x_asc = np.log(x_asc)
    w_asc = np.clip(w_matrix[:, ::-1], 0.0, 1.0)
    w_asc = w_asc / np.clip(np.sum(w_asc, axis=1, keepdims=True), 1e-12, None)
    q3_asc = np.cumsum(w_asc, axis=1)

    d10, d50, d90 = np.zeros(len(w_matrix)), np.zeros(len(w_matrix)), np.zeros(len(w_matrix))
    for k in range(len(w_matrix)):
        cdf_k = q3_asc[k]
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

    sec_per_min = 60.0
    C_mill = cfg.rho_slurry * cfg.Cp_slurry * cfg.V_m
    C_tank = cfg.rho_slurry * cfg.Cp_slurry * cfg.V_t

    def rhs(t, y):
        w_m = np.clip(y[:N], 0.0, None)
        w_m = w_m / np.sum(w_m)
        w_t = np.clip(y[N:2*N], 0.0, None)
        w_t = w_t / np.sum(w_t)
        T_m = max(float(y[2*N]), 273.15)
        T_t = max(float(y[2*N + 1]), 273.15)

        inv_T_diff = (1.0 / T_m) - (1.0 / cfg.T_ref)
        mu_solvent = (cfg.mu_ref / crowd_factor) * np.exp(cfg.E_mu_over_R * inv_T_diff)
        d32_m = 1.0 / np.sum(w_m / x)
        area_ratio = max(d32_init / d32_m - 1.0, 0.0)
        mu_eff = mu_solvent * crowd_factor * np.exp(kappa_A * area_ratio)

        S0_t = (
            S0_ref
            * np.exp(-Ea_over_R * inv_T_diff)
            * ((cfg.mu_ref / max(mu_eff, 1e-5)) ** nu_mu)
        )

        dw_m_dt = (1.0 / cfg.tau_m) * (w_t - w_m) + S0_t * (K_tilde @ w_m)
        dw_t_dt = (1.0 / cfg.tau_t) * (w_m - w_t)

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


def compute_kwade_stress_scaling(
    d_gm_um: float,
    v_tip_ms: float,
    phi_gm: float,
    rho_gm_kgm3: float,
    SE_opt_mJ: float,
    d_gm_lab_um: float = 400.0,
    v_tip_lab_ms: float = 10.0,
    phi_gm_lab: float = 0.80,
    rho_gm_lab_kgm3: float = 6050.0,
    sigma_se: float = 0.85,
    eps_pack: float = 0.38,
) -> Dict[str, float]:
    d_m = d_gm_um * 1e-6
    d_lab_m = d_gm_lab_um * 1e-6
    SE_pilot_mJ = (d_m ** 3) * (v_tip_ms ** 2) * rho_gm_kgm3 * 1e3
    SE_lab_mJ = (d_lab_m ** 3) * (v_tip_lab_ms ** 2) * rho_gm_lab_kgm3 * 1e3

    def calc_eta_se(se_val: float) -> float:
        log_ratio = np.log(max(se_val, 1e-9) / max(SE_opt_mJ, 1e-9))
        gauss_term = np.exp(-0.5 * ((log_ratio / max(sigma_se, 0.1)) ** 2))
        excess_penalty = (se_val / SE_opt_mJ) ** (-0.25) if se_val > SE_opt_mJ else 1.0
        return float(gauss_term * excess_penalty)

    eta_pilot = calc_eta_se(SE_pilot_mJ)
    eta_lab = calc_eta_se(SE_lab_mJ)

    g_pilot = (phi_gm * (1.0 - phi_gm)) / ((1.0 - phi_gm * (1.0 - eps_pack)) ** 2)
    g_lab = (phi_gm_lab * (1.0 - phi_gm_lab)) / ((1.0 - phi_gm_lab * (1.0 - eps_pack)) ** 2)

    power_density_ratio = (
        ((v_tip_ms / max(v_tip_lab_ms, 1e-3)) ** 2.75)
        * ((phi_gm / phi_gm_lab) ** 1.35)
        * ((rho_gm_kgm3 / rho_gm_lab_kgm3) ** 0.85)
        * ((d_gm_um / d_gm_lab_um) ** 0.15)
    )

    rel_stress_freq = (
        power_density_ratio * (SE_lab_mJ / max(SE_pilot_mJ, 1e-9)) * (g_pilot / g_lab)
    )
    S0_multiplier = power_density_ratio * (eta_pilot / max(eta_lab, 1e-6)) * (g_pilot / g_lab)

    return {
        "SE_pilot_mJ": SE_pilot_mJ,
        "SE_lab_mJ": SE_lab_mJ,
        "SE_ratio_to_opt": SE_pilot_mJ / max(SE_opt_mJ, 1e-9),
        "eta_pilot": eta_pilot,
        "eta_lab": eta_lab,
        "power_density_ratio": power_density_ratio,
        "rel_stress_freq": rel_stress_freq,
        "S0_multiplier": S0_multiplier,
    }


def compute_thermal_and_energy_metrics(
    cfg: MillLoopConfig, sim: Dict[str, np.ndarray], w_solids_mass_frac: float = 0.24
) -> Dict[str, np.ndarray]:
    t_sec = sim["t"] * 60.0
    P_shaft_W = cfg.P_agit_W * ((sim["mu_eff"] / cfg.mu_ref) ** cfg.alpha_P)
    Q_gen_W = cfg.eta_diss * P_shaft_W
    Q_jacket_mill_W = cfg.UA_mill * (sim["T_m"] - cfg.T_cool_m)
    Q_jacket_tank_W = cfg.UA_tank * (sim["T_t"] - cfg.T_cool_t)

    E_cum_J = np.zeros_like(t_sec)
    if len(t_sec) > 1:
        dt_sec = np.diff(t_sec)
        E_cum_J[1:] = np.cumsum(0.5 * (P_shaft_W[:-1] + P_shaft_W[1:]) * dt_sec)

    total_slurry_kg = (cfg.rho_slurry / 1000.0) * (cfg.V_m + cfg.V_t)
    total_solids_kg = max(total_slurry_kg * w_solids_mass_frac, 1e-6)
    E_specific_kWh_kg = E_cum_J / (3.6e6 * total_solids_kg)

    return {
        "P_shaft_kW": P_shaft_W / 1000.0,
        "Q_gen_kW": Q_gen_W / 1000.0,
        "Q_mill_kW": Q_jacket_mill_W / 1000.0,
        "Q_tank_kW": Q_jacket_tank_W / 1000.0,
        "Q_total_cool_kW": (Q_jacket_mill_W + Q_jacket_tank_W) / 1000.0,
        "E_specific_kWh_kg": E_specific_kWh_kg,
    }


def solve_required_pilot_cooling(
    pilot_base_cfg: MillLoopConfig,
    scaled_params: Dict[str, float],
    w0: np.ndarray,
    t_eval: np.ndarray,
    T_max_limit_K: float,
) -> Dict[str, float]:
    sim_baseline = simulate_coupled_mill_tank_loop(pilot_base_cfg, scaled_params, w0, t_eval)
    peak_T_geom = float(np.max(sim_baseline["T_m"]))

    if peak_T_geom <= T_max_limit_K:
        return {
            "UA_mill_req": pilot_base_cfg.UA_mill,
            "UA_supplemental_HX": 0.0,
            "peak_T_geom_C": peak_T_geom - 273.15,
            "needs_supplemental_cooling": False,
        }

    ua_low = pilot_base_cfg.UA_mill
    ua_high = pilot_base_cfg.UA_mill * 18.0
    for _ in range(16):
        ua_mid = 0.5 * (ua_low + ua_high)
        test_cfg = MillLoopConfig(**{**pilot_base_cfg.__dict__, "UA_mill": ua_mid})
        sim_test = simulate_coupled_mill_tank_loop(test_cfg, scaled_params, w0, t_eval)
        if float(np.max(sim_test["T_m"])) > T_max_limit_K:
            ua_low = ua_mid
        else:
            ua_high = ua_mid

    return {
        "UA_mill_req": ua_high,
        "UA_supplemental_HX": max(ua_high - pilot_base_cfg.UA_mill, 0.0),
        "peak_T_geom_C": peak_T_geom - 273.15,
        "needs_supplemental_cooling": True,
    }


# =====================================================================
# 5. PYTORCH THERMO-VISCOUS PBE KERNEL & DUAL-STATE PINN MODULES
# =====================================================================
class ThermoViscousPBEKernels(nn.Module):
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

        inv_d32 = torch.matmul(w_m, self.inv_x.unsqueeze(-1))
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
    def __init__(self, n_bins: int, t_max: float, T_ref: float = 298.15, hidden_dim: int = 64):
        super().__init__()
        self.n_bins = n_bins
        self.t_max = t_max
        self.T_ref = T_ref
        self.T_scale = 25.0

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
        self.head_T = nn.Linear(hidden_dim, 2)

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
# 6. COMPOSITE PHYSICS + DATA LOSS & FISHER INFORMATION IDENTIFIABILITY
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

    w_m_c, w_t_c, T_m_c, T_t_c, dw_m_dt, dw_t_dt, dT_m_dt, dT_t_dt = (
        pinn.forward_with_jvp_time_derivatives(t_colloc)
    )

    res_tank_psd = cfg.tau_t * dw_t_dt - (w_m_c - w_t_c)
    loss_tank_pbe = torch.mean(res_tank_psd ** 2)

    breakage_rate, _, mu_eff_c = kernels.compute_mill_breakage_rate(w_m_c, T_m_c)
    res_mill_psd = cfg.tau_m * dw_m_dt - (w_t_c - w_m_c) - cfg.tau_m * breakage_rate
    loss_mill_pbe = torch.mean(res_mill_psd ** 2)

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
# SIDEBAR: NETZSCH DELTAVITA LAB MILL, AGITATOR RPM & MATERIAL CONFIG
# =====================================================================
user_consts = get_const_dict()

st.sidebar.title("⚙️ NETZSCH DeltaVita® Config")

with st.sidebar.expander("1. Lab Mill, Agitator RPM, Liner & Beads", expanded=True):
    lab_model_key = st.selectbox(
        "Lab Reference NETZSCH Model",
        options=list(NETZSCH_DELTAVITA_MODELS.keys()),
        index=1,  # Default: DV 300 (0.30 L)
    )
    lab_chamber_key = st.selectbox(
        "Lab Grinding Chamber Liner",
        options=list(CHAMBER_LINER_MATERIALS.keys()),
        index=1,  # Default: NETZSCH-Ceram C (SiC)
    )
    lab_bead_key = st.selectbox(
        "Lab Grinding Media (Bead) Type",
        options=list(BEAD_MEDIA_TYPES.keys()),
        index=0,  # Default: VitaBeaDs NaNo (Y-ZrO2)
    )

    lab_m_spec = NETZSCH_DELTAVITA_MODELS[lab_model_key]
    lab_c_spec = CHAMBER_LINER_MATERIALS[lab_chamber_key]
    lab_b_spec = BEAD_MEDIA_TYPES[lab_bead_key]
    UA_m_calc, UA_t_calc, U_m_coeff = compute_netzsch_chamber_ua(
        lab_model_key,
        lab_chamber_key,
        h_slurry_Wm2K=user_consts["h_slurry"],
        h_coolant_Wm2K=user_consts["h_coolant"],
        U_tank_Wm2K=user_consts["U_tank"],
    )

    max_rpm_lab = int(tip_speed_to_rpm(lab_c_spec["max_tip_speed_ms"], lab_m_spec["D_rotor_mm"]))
    nom_rpm_lab = int(lab_m_spec["rpm_nominal_10ms"])

    lab_rpm_in = st.slider(
        f"Lab Agitator Speed (RPM) [D_rot={lab_m_spec['D_rotor_mm']:.0f}mm]",
        min_value=800,
        max_value=max_rpm_lab,
        value=min(nom_rpm_lab, max_rpm_lab),
        step=25,
    )
    v_tip_lab_ms = rpm_to_tip_speed(lab_rpm_in, lab_m_spec["D_rotor_mm"])
    lab_d_gm_um = st.slider("Lab Bead Diameter d_GM (µm)", 100.0, 1000.0, 400.0, 25.0)
    lab_phi_gm = st.slider("Lab Bead Filling Ratio φ_GM (-)", 0.65, 0.90, 0.80, 0.01)
    st.caption(
        f"⚡ **Lab Tip Speed v_t = {v_tip_lab_ms:.2f} m/s** "
        f"(Nominal 10 m/s = {nom_rpm_lab} RPM)"
    )

    V_m_in = st.number_input("Active Chamber Volume V_m (L)", 0.05, 5.00, float(lab_m_spec["V_m"]), 0.05)
    V_t_in = st.slider("Holding Tank Volume V_t (L)", 0.50, 30.00, float(lab_m_spec["V_t_default"]), 0.25)
    Q_in = st.slider("Recirculation Rate Q (L/min)", 0.10, 10.00, float(lab_m_spec["Q_default"]), 0.05)
    t_max_in = st.slider("Batch Grinding Duration (min)", 20.0, 180.0, 60.0, 5.0)

with st.sidebar.expander("2. Thermal & Operating Setpoints", expanded=False):
    P_lab_default = (
        lab_m_spec["P_agit_W"]
        * ((v_tip_lab_ms / 10.0) ** 2.75)
        * ((lab_b_spec["rho_kgm3"] / 6050.0) ** 0.85)
    )
    P_agit_in = st.slider(
        "Shaft Power P_0 (W) [Auto-scaled by RPM]",
        50.0, 6000.0, float(np.clip(P_lab_default, 50.0, 6000.0)), 10.0
    )
    UA_mill_in = st.slider("Mill Jacket UA (W/K) [Auto-computed]", 1.0, 250.0, float(round(UA_m_calc, 1)), 0.5)
    UA_tank_in = st.slider("Tank Jacket UA (W/K) [Auto-computed]", 2.0, 300.0, float(round(UA_t_calc, 1)), 1.0)
    T_cool_C = st.slider("Chiller Supply Temp (°C)", 5.0, 25.0, 15.0, 1.0)
    phi_v_in = st.slider("Solids Volume Fraction φ_v", 0.05, 0.40, 0.20, 0.01)
    st.caption(f"Computed Liner U_m = **{U_m_coeff:.0f} W/(m²·K)** | Bead ρ = **{lab_b_spec['rho_kgm3']:.0f} kg/m³**")

with st.sidebar.expander("3. Ground-Truth PBE Kernel Parameters", expanded=True):
    S0_base_10ms = st.slider("Base S0_ref at 10 m/s (min⁻¹)", 0.15, 1.50, 0.65, 0.05)
    alpha_true = st.slider("True Selection Exponent α", 0.50, 2.00, 1.15, 0.05)
    phi_true = st.slider("True Bimodal Weight φ", 0.20, 0.80, 0.55, 0.05)
    gamma_true = st.slider("True Fine Progeny Modulus γ", 0.35, 1.40, 0.75, 0.05)
    beta_true = st.slider("True Coarse Progeny Modulus β", 1.50, 4.00, 2.50, 0.10)
    nu_mu_true = st.slider("True Viscous Damping Exponent ν_μ", 0.10, 1.00, 0.45, 0.05)
    kappa_A_true = st.slider("True Fines Thickening Factor κ_A", 0.05, 0.60, 0.22, 0.02)

    lab_kwade_mod = compute_kwade_stress_scaling(
        d_gm_um=lab_d_gm_um,
        v_tip_ms=v_tip_lab_ms,
        phi_gm=lab_phi_gm,
        rho_gm_kgm3=lab_b_spec["rho_kgm3"],
        SE_opt_mJ=0.035,
        d_gm_lab_um=400.0,
        v_tip_lab_ms=10.0,
        phi_gm_lab=0.80,
        rho_gm_lab_kgm3=6050.0,
        sigma_se=user_consts["sigma_se"],
        eps_pack=user_consts["eps_pack"],
    )
    S0_true = float(S0_base_10ms * lab_kwade_mod["S0_multiplier"])
    st.caption(f"Effective Lab **S0_ref = {S0_true:.3f} min⁻¹** at **{lab_rpm_in} RPM** ({v_tip_lab_ms:.2f} m/s)")

cfg = MillLoopConfig(
    V_m=V_m_in,
    V_t=V_t_in,
    Q=Q_in,
    t_max=t_max_in,
    rho_slurry=user_consts["rho_slurry"],
    Cp_slurry=user_consts["Cp_slurry"],
    phi_v=phi_v_in,
    phi_max=user_consts["phi_max"],
    T_ref=user_consts["T_ref"],
    mu_ref=user_consts["mu_ref"],
    E_mu_over_R=user_consts["E_mu_over_R"],
    P_agit_W=P_agit_in,
    eta_diss=user_consts["eta_diss"],
    alpha_P=user_consts["alpha_P"],
    UA_mill=UA_mill_in,
    UA_tank=UA_tank_in,
    T_cool_m=273.15 + T_cool_C,
    T_cool_t=273.15 + T_cool_C + 2.0,
    T_init=user_consts["T_ref"],
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
st.title("⚙️ NETZSCH DeltaVita® Thermo-Viscous Recirculation PBE-PINN & 3D Loop Digital Twin")
st.markdown(
    "Deconvolve **holding-tank residence-time lag** ($\tau_t = V_t/Q$), **non-isothermal chamber wall heat transfer (316L SS vs. Ceramic)**, "
    "and **agitator RPM / tip-speed stress energy** across **NETZSCH DV 200, DV 300, DV 600, and DV 2000**."
)

kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
kpi1.metric("Lab Agitator Speed", f"{lab_rpm_in} RPM", f"v_t = {v_tip_lab_ms:.2f} m/s ({lab_model_key[:6]})")
kpi2.metric("Mill Residence (τ_m)", f"{cfg.tau_m:.2f} min", f"V_m = {cfg.V_m:.2f} L")
kpi3.metric("Tank Lag Time (τ_t)", f"{cfg.tau_t:.2f} min", f"V_t/V_m = {cfg.V_t/cfg.V_m:.1f}x")
max_d90_lag = float(np.max(sim_data["d90_t"] - sim_data["d90_m"]))
kpi4.metric("Peak Tank–Mill D90 Lag", f"{max_d90_lag:.2f} µm", "Tank coarser than mill")
peak_dT = float(np.max(sim_data["T_m"] - sim_data["T_t"]))
kpi5.metric("Peak Mill–Tank ΔT", f"+{peak_dT:.2f} °C", f"Max T_m = {np.max(sim_data['T_m'])-273.15:.1f} °C")
kpi6.metric("Final Holding Tank D90", f"{sim_data['d90_t'][-1]:.2f} µm", f"D32 = {sim_data['d32_t'][-1]:.2f} µm")

# =====================================================================
# WORKBENCH TABS (6 TABS — TAB 0 IS LANDING THEORY, CONSTANTS & 3D TWIN)
# =====================================================================
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "0️⃣ Theory, Equations, Constants & 3D Loop Digital Twin",
    "1️⃣ Forward Digital Twin & Lag Explorer",
    "2️⃣ Experimental Tank Sampler & CSV I/O",
    "3️⃣ 3-Stage Curriculum Inverse PINN",
    "4️⃣ Deconvolved Mill vs. Tank Diagnostics",
    "5️⃣ NETZSCH DV200–DV2000 Scale-Up, RPM & Kwade Optimizer",
])

# ---------------------------------------------------------------------
# TAB 0: ASSUMPTIONS, FORMULAE, EDITABLE CONSTANTS & 3D LOOP DIGITAL TWIN
# ---------------------------------------------------------------------
with tab0:
    st.subheader("🌐 Interactive 3D CAD Digital Twin: Closed-Loop Recirculation Skid & NETZSCH Grinding Chamber")
    c_3d_1, c_3d_2, c_3d_3, c_3d_4 = st.columns([1.3, 1.3, 1.2, 1.1])
    viz_scope = c_3d_1.selectbox(
        "3D Digital Twin View Mode",
        options=[
            "Full Recirculation Loop (Tank + Pump + HX + Mill)",
            "Mill Chamber Close-Up Only",
        ],
        index=0,
    )
    viz_model_key = c_3d_2.selectbox(
        "Inspect NETZSCH Mill Geometry in 3D",
        options=list(NETZSCH_DELTAVITA_MODELS.keys()),
        index=list(NETZSCH_DELTAVITA_MODELS.keys()).index(lab_model_key),
    )
    viz_color_mode = c_3d_3.radio(
        "3D Bead Cloud Color Field",
        options=["Local Stress Energy SE_GM (mJ)", "Local Slurry Temperature (°C)"],
        horizontal=False,
    )
    viz_time_min = c_3d_4.slider(
        "Process Timestamp t (min)",
        0.0, float(cfg.t_max), float(min(15.0, cfg.t_max * 0.4)), 1.0
    )

    idx_3d = int(np.argmin(np.abs(sim_data["t"] - viz_time_min)))
    T_tank_3d_C = float(sim_data["T_t"][idx_3d] - 273.15)
    T_mill_3d_C = float(sim_data["T_m"][idx_3d] - 273.15)
    d90_tank_3d = float(sim_data["d90_t"][idx_3d])
    d90_mill_3d = float(sim_data["d90_m"][idx_3d])
    viz_m_spec = NETZSCH_DELTAVITA_MODELS[viz_model_key]
    viz_rpm = tip_speed_to_rpm(v_tip_lab_ms, viz_m_spec["D_rotor_mm"])

    fig_3d = build_3d_recirculation_loop_digital_twin(
        model_key=viz_model_key,
        chamber_key=lab_chamber_key,
        bead_key=lab_bead_key,
        rpm=viz_rpm,
        d_gm_um=lab_d_gm_um,
        phi_gm=lab_phi_gm,
        Q_flow_Lmin=cfg.Q,
        T_tank_C=T_tank_3d_C,
        T_mill_C=T_mill_3d_C,
        T_cool_C=T_cool_C,
        d90_tank_um=d90_tank_3d,
        d90_mill_um=d90_mill_3d,
        SE_max_mJ=lab_kwade_mod["SE_pilot_mJ"],
        color_mode=viz_color_mode,
        view_scope=viz_scope,
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    st.divider()

    # SECTION B: FUNDAMENTAL ASSUMPTIONS & MATHEMATICAL FORMULAE
    st.subheader("📐 Fundamental Physical Assumptions & Governing Mathematical Formulae")
    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.markdown("#### 1. Core Process & Transport Assumptions")
        st.markdown(
            """
            1. **Dual-Control-Volume Recirculation Architecture:** The active grinding chamber ($V_m$) and recirculation holding tank ($V_t$) are modeled as coupled, well-mixed control volumes connected by volumetric flow rate $Q$, with mean residence times $\\tau_m = V_m/Q$ and $\\tau_t = V_t/Q$.
            2. **Zero Comminution in the Holding Tank:** Particle breakage occurs exclusively within the bead-filled mill chamber ($V_m$). The holding tank acts as a non-reactive convective buffer that introduces residence-time lag and capacitance smoothing between the mill outlet and experimental sampling port.
            3. **Hard Simplex Mass Conservation:** Discrete particle size classes $x_1 > x_2 > \\dots > x_N$ conserve total solid mass strictly ($\\sum_{i=1}^N w_{m,i}(t) = \\sum_{i=1}^N w_{t,i}(t) \\equiv 1$) via Softmax output projection and column-conservative kernel assembly ($\\mathbf{1}^\\top \\widetilde{\\mathbf{K}} = \\mathbf{0}^\\top$).
            4. **Opposing Thermo-Viscous Rheology:** Slurry viscosity $\\mu_{\\text{eff}}(T_m, \\mathbf{w}_m)$ couples continuous-phase Andrade thermal thinning against Krieger–Dougherty solids packing and specific surface area ($D_{32}^{-1}$) fines-thickening.
            5. **1D Radial-Composite Fourier Wall Heat Transfer:** Jacket heat removal accounts for internal agitated slurry convection ($h_{\\text{slurry}}$), conductive resistance across the specific liner thickness ($\\delta_{\\text{liner}} / k_{\\text{liner}}$ for 316L SS, $\\text{SiC}$, $\\text{Si}_3\\text{N}_4$, or $\\text{ZrO}_2$), and external coolant convection ($h_{\\text{coolant}}$).
            """
        )

        st.markdown("#### 2. Coupled Mill–Tank Population Balance Equations (PBE)")
        st.latex(
            r"\tau_m \frac{d\hat{\mathbf{w}}_m}{dt} = \left(\hat{\mathbf{w}}_t - \hat{\mathbf{w}}_m\right) + \tau_m \, S_0\!\left(\hat{T}_m, \hat{\mu}_{\text{eff}}\right) \widetilde{\mathbf{K}}(\alpha, \phi, \gamma, \beta)\,\hat{\mathbf{w}}_m"
        )
        st.latex(
            r"\tau_t \frac{d\hat{\mathbf{w}}_t}{dt} = \hat{\mathbf{w}}_m - \hat{\mathbf{w}}_t \quad \Longleftrightarrow \quad \hat{\mathbf{w}}_m(t) = \hat{\mathbf{w}}_t(t) + \tau_t \frac{d\hat{\mathbf{w}}_t}{dt}"
        )
        st.latex(
            r"\widetilde{\mathbf{K}} = (\mathbf{b} - \mathbf{I})\operatorname{diag}\!\left(\hat{\mathbf{s}}(\alpha)\right), \qquad \hat{s}_j = \left(\frac{x_j}{x_1}\right)^\alpha, \qquad B_{ij} = \phi\left(\frac{x_{i-1}}{x_j}\right)^\gamma + (1-\phi)\left(\frac{x_{i-1}}{x_j}\right)^\beta"
        )

    with col_a2:
        st.markdown("#### 3. Non-Isothermal Energy Balances & Composite Liner Conductance")
        st.latex(
            r"\rho_s C_p V_m \frac{d\hat{T}_m}{dt} = \rho_s C_p Q\left(\hat{T}_t - \hat{T}_m\right) + \eta_{\text{diss}} P_0 \left(\frac{\hat{\mu}_{\text{eff}}}{\mu_{\text{ref}}}\right)^{\alpha_P} - U_m A_m \left(\hat{T}_m - T_{\text{cool},m}\right)"
        )
        st.latex(
            r"\rho_s C_p V_t \frac{d\hat{T}_t}{dt} = \rho_s C_p Q\left(\hat{T}_m - \hat{T}_t\right) - U_t A_t \left(\hat{T}_t - T_{\text{cool},t}\right)"
        )
        st.latex(
            r"U_m = \left(\frac{1}{h_{\text{slurry}}} + \frac{\delta_{\text{liner}}}{k_{\text{liner}}} + \frac{1}{h_{\text{coolant}}}\right)^{-1}"
        )

        st.markdown("#### 4. Slurry Rheology, Kwade Stress Energy & Agitator RPM Scaling")
        st.latex(
            r"\hat{\mu}_{\text{eff}}\!\left(\hat{T}_m, \hat{\mathbf{w}}_m\right) = \mu_{L,\text{ref}}\exp\!\left[\frac{E_\mu}{R}\left(\frac{1}{\hat{T}_m} - \frac{1}{T_{\text{ref}}}\right)\right]\left(1 - \frac{\phi_v}{\phi_{\max}}\right)^{-2.5\phi_{\max}}\exp\!\left[\kappa_A\left(\frac{D_{32,0}}{D_{32}(\hat{\mathbf{w}}_m)} - 1\right)\right]"
        )
        st.latex(
            r"v_t = \frac{\pi D_{\text{rotor}} N_{\text{RPM}}}{60}, \qquad SE_{GM} = d_{GM}^3 \, v_t^2 \, \rho_{GM}, \qquad S_0(t) = S_{0,\text{ref}}^{\text{lab}} \frac{(P_0/V_m)}{(P_0/V_m)_{\text{lab}}} \frac{\eta_{SE}(SE_{GM})}{\eta_{SE}(SE_{GM,\text{lab}})} \frac{g(\phi_{GM})}{g(\phi_{GM,\text{lab}})} \left(\frac{\mu_{\text{ref}}}{\hat{\mu}_{\text{eff}}}\right)^{\nu_\mu}"
        )

    st.divider()

    # SECTION C: EDITABLE PHYSICAL & MODEL CONSTANTS TABLE
    st.subheader("🧮 Editable Physical, Thermal & Rheological Constants Table")
    st.markdown(
        "Double-click any cell in the **Value** column below to modify a constant. "
        "All simulations across Tabs 0–5 automatically update using your edited values."
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    if col_btn1.button("🔄 Reset Constants to Defaults", use_container_width=True):
        st.session_state["custom_constants_df"] = pd.DataFrame(DEFAULT_CONSTANTS_ROWS)
        st.rerun()

    edited_df = st.data_editor(
        st.session_state["custom_constants_df"],
        disabled=["Symbol", "Parameter Name", "Unit", "Physical Role"],
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key="constants_editor_widget",
    )

    if not edited_df["Value"].equals(st.session_state["custom_constants_df"]["Value"]):
        st.session_state["custom_constants_df"] = edited_df.copy()
        st.rerun()

# ---------------------------------------------------------------------
# TAB 1: FORWARD DIGITAL TWIN & LAG EXPLORER
# ---------------------------------------------------------------------
with tab1:
    st.subheader(
        f"Coupled Mill Chamber vs. Holding Tank Trajectories ({lab_model_key[:6]} @ {lab_rpm_in} RPM / {v_tip_lab_ms:.2f} m/s)"
    )
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

    fig_fwd.add_trace(go.Scatter(x=sim_data["t"], y=sim_data["d90_t"], name="Tank D90 (Sampled)", line=dict(color="#1f77b4", width=3)), row=1, col=1)
    fig_fwd.add_trace(go.Scatter(x=sim_data["t"], y=sim_data["d90_m"], name="Mill D90 (Latent)", line=dict(color="#ff7f0e", width=2.5, dash="dash")), row=1, col=1)
    fig_fwd.add_trace(go.Scatter(x=sim_data["t"], y=sim_data["d50_t"], name="Tank D50", line=dict(color="#2ca02c", width=2)), row=1, col=1)
    fig_fwd.add_trace(go.Scatter(x=sim_data["t"], y=sim_data["d50_m"], name="Mill D50", line=dict(color="#d62728", width=2, dash="dot")), row=1, col=1)

    fig_fwd.add_trace(go.Scatter(x=sim_data["t"], y=sim_data["T_m"] - 273.15, name="Mill Chamber T_m (°C)", line=dict(color="#d62728", width=3)), row=1, col=2)
    fig_fwd.add_trace(go.Scatter(x=sim_data["t"], y=sim_data["T_t"] - 273.15, name="Holding Tank T_t (°C)", line=dict(color="#1f77b4", width=3, dash="dash")), row=1, col=2)

    fig_fwd.add_trace(go.Scatter(x=sim_data["t"], y=sim_data["mu_eff"] * 1e3, name="Slurry Viscosity μ_eff (mPa·s)", line=dict(color="#9467bd", width=3)), row=2, col=1, secondary_y=False)
    fig_fwd.add_trace(go.Scatter(x=sim_data["t"], y=sim_data["S0_t"], name="Dynamic Selection Rate S0(t) (min⁻¹)", line=dict(color="#2ca02c", width=2.5, dash="dot")), row=2, col=1, secondary_y=True)

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

        init_guess = {
            "S0_ref": max(0.20, S0_true * 0.65),
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
            "lab_rpm_ref": lab_rpm_in,
            "lab_v_tip_ref": v_tip_lab_ms,
            "lab_d_gm_ref": lab_d_gm_um,
        }

    if "pinn_results" in st.session_state:
        res = st.session_state["pinn_results"]
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
        fig_diag.add_trace(go.Scatter(x=df_obs["time_min"], y=d90_obs, mode="markers", name="Noisy Tank Samples D90", marker=dict(size=9, color="#1f77b4", symbol="circle-open", line=dict(width=2))), row=1, col=1)
        fig_diag.add_trace(go.Scatter(x=sim_data["t"], y=d90_t_p, name="PINN Tank D90_t(t)", line=dict(color="#1f77b4", width=3)), row=1, col=1)
        fig_diag.add_trace(go.Scatter(x=sim_data["t"], y=sim_data["d90_m"], name="True Latent Mill D90_m(t)", line=dict(color="black", width=2, dash="dot")), row=1, col=1)
        fig_diag.add_trace(go.Scatter(x=sim_data["t"], y=d90_m_p, name="PINN Deconvolved Mill D90_m(t)", line=dict(color="#ff7f0e", width=3, dash="dash")), row=1, col=1)

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

# ---------------------------------------------------------------------
# TAB 5: NETZSCH DELTAVITA (DV200/300/600/2000), AGITATOR RPM & KWADE OPTIMIZER
# ---------------------------------------------------------------------
with tab5:
    st.subheader("🚀 NETZSCH DeltaVita® (DV 200 / DV 300 / DV 600 / DV 2000) Scale-Up, Agitator RPM & Kwade Optimizer")

    if "pinn_results" in st.session_state:
        base_k_params = st.session_state["pinn_results"]["learned_params"]
        ref_v_tip_lab = st.session_state["pinn_results"].get("lab_v_tip_ref", v_tip_lab_ms)
        ref_d_gm_lab = st.session_state["pinn_results"].get("lab_d_gm_ref", lab_d_gm_um)
        st.success(
            f"✅ **Active Kernel Source:** Using **PINN-Identified Thermo-Viscous Parameters** from `{lab_model_key}` "
            f"(S0_ref = {base_k_params['S0_ref']:.3f} min⁻¹ at {ref_v_tip_lab:.2f} m/s, α = {base_k_params['alpha']:.2f}, "
            f"γ = {base_k_params['gamma']:.2f}, ν_μ = {base_k_params['nu_mu']:.2f}, κ_A = {base_k_params['kappa_A']:.2f})."
        )
    else:
        base_k_params = true_params_dict
        ref_v_tip_lab = v_tip_lab_ms
        ref_d_gm_lab = lab_d_gm_um
        st.info(
            f"ℹ️ **Active Kernel Source:** Using baseline parameters from `{lab_model_key}` "
            f"({lab_rpm_in} RPM / {v_tip_lab_ms:.2f} m/s). Run Tab 3 to lock onto inverse PINN parameters."
        )

    with st.expander("🏭 1. Select Target NETZSCH DeltaVita® Model, Chamber Liner (SS vs. Ceramic) & Bead Media", expanded=True):
        m_c1, m_c2, m_c3 = st.columns(3)
        target_model_key = m_c1.selectbox(
            "Target NETZSCH DeltaVita® Model",
            options=list(NETZSCH_DELTAVITA_MODELS.keys()),
            index=3,
        )
        target_chamber_key = m_c2.selectbox(
            "Target Grinding Chamber Material (SS vs. Ceramic)",
            options=list(CHAMBER_LINER_MATERIALS.keys()),
            index=0,
        )
        target_bead_key = m_c3.selectbox(
            "Target Grinding Bead Material (Zirconium vs. Ceramic)",
            options=list(BEAD_MEDIA_TYPES.keys()),
            index=0,
        )

        t_model_spec = NETZSCH_DELTAVITA_MODELS[target_model_key]
        t_chamber_spec = CHAMBER_LINER_MATERIALS[target_chamber_key]
        t_bead_spec = BEAD_MEDIA_TYPES[target_bead_key]
        UA_m_target, UA_t_target, U_m_target = compute_netzsch_chamber_ua(
            target_model_key,
            target_chamber_key,
            h_slurry_Wm2K=user_consts["h_slurry"],
            h_coolant_Wm2K=user_consts["h_coolant"],
            U_tank_Wm2K=user_consts["U_tank"],
        )

        c_su1, c_su2, c_su3, c_su4 = st.columns(4)
        V_m_pilot = c_su1.number_input("Target Chamber Volume V_m (L)", 0.10, 20.0, float(t_model_spec["V_m"]), 0.10)
        V_t_pilot = c_su2.number_input("Target Holding Tank Volume V_t (L)", 0.50, 200.0, float(t_model_spec["V_t_default"]), 0.50)
        Q_pilot = c_su3.number_input("Target Recirculation Flow Q (L/min)", 0.20, 50.0, float(t_model_spec["Q_default"]), 0.20)
        t_max_pilot = c_su4.number_input("Batch Simulation Horizon (min)", 20.0, 360.0, 90.0, 10.0)

    with st.expander("🔄 2. Agitator Speed Control (Direct RPM or Tip Speed v_t) & Kwade Bead Parameters", expanded=True):
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.3, 1.7, 1.3])
        speed_ctrl_mode = ctrl_col1.radio(
            "Agitator Speed Control Mode",
            options=["Set by Agitator RPM", "Set by Tip Speed v_t (m/s)"],
            horizontal=False,
        )

        d_rotor_target_mm = t_model_spec["D_rotor_mm"]
        max_v_tip_allowed = float(t_chamber_spec["max_tip_speed_ms"])
        min_rpm_target = int(tip_speed_to_rpm(5.0, d_rotor_target_mm))
        max_rpm_target = int(tip_speed_to_rpm(max_v_tip_allowed, d_rotor_target_mm))
        default_rpm_target = int(np.clip(tip_speed_to_rpm(10.5, d_rotor_target_mm), min_rpm_target, max_rpm_target))

        if speed_ctrl_mode == "Set by Agitator RPM":
            pilot_rpm_selected = ctrl_col2.slider(
                f"Target Agitator Shaft Speed (RPM) [Rotor Ø = {d_rotor_target_mm:.0f} mm]",
                min_value=min_rpm_target,
                max_value=max_rpm_target,
                value=default_rpm_target,
                step=25,
            )
            v_tip_pilot_ms = rpm_to_tip_speed(pilot_rpm_selected, d_rotor_target_mm)
        else:
            v_tip_pilot_ms = ctrl_col2.slider(
                f"Target Peripheral Tip Speed v_t (m/s) [Rotor Ø = {d_rotor_target_mm:.0f} mm]",
                min_value=5.0,
                max_value=max_v_tip_allowed,
                value=min(10.5, max_v_tip_allowed),
                step=0.25,
            )
            pilot_rpm_selected = int(round(tip_speed_to_rpm(v_tip_pilot_ms, d_rotor_target_mm)))

        contour_y_axis_mode = ctrl_col3.radio(
            "2D Contour Plot Y-Axis Unit",
            options=["Agitator Speed (RPM)", "Tip Speed v_t (m/s)"],
            index=0,
        )
        ctrl_col3.caption(
            f"📌 **Coupled Readout:** **{pilot_rpm_selected} RPM** ⟺ **{v_tip_pilot_ms:.2f} m/s**"
        )

        k_c1, k_c2, k_c3, k_c4, k_c5 = st.columns(5)
        d_gm_pilot_um = k_c1.slider("Pilot Bead Diameter d_GM (µm)", 100.0, 1000.0, 300.0, 25.0)
        phi_gm_pilot = k_c2.slider("Bead Filling Ratio φ_GM (-)", 0.65, 0.90, 0.82, 0.01)
        SE_opt_mJ = k_c3.number_input("API Optimal SE_opt (mJ)", 0.005, 1.500, 0.035, 0.005, format="%.3f")
        d90_spec_um = k_c4.slider("Target Product D90 (µm)", 0.20, 3.00, 0.55, 0.05)
        T_max_limit_C = k_c5.slider("Max Slurry Temp Limit (°C)", 25.0, 55.0, 35.0, 1.0)

        T_cool_pilot_C = 12.0
        max_supp_UA = 150.0

    kwade_vs_lab = compute_kwade_stress_scaling(
        d_gm_um=d_gm_pilot_um,
        v_tip_ms=v_tip_pilot_ms,
        phi_gm=phi_gm_pilot,
        rho_gm_kgm3=t_bead_spec["rho_kgm3"],
        SE_opt_mJ=SE_opt_mJ,
        d_gm_lab_um=ref_d_gm_lab,
        v_tip_lab_ms=ref_v_tip_lab,
        phi_gm_lab=lab_phi_gm,
        rho_gm_lab_kgm3=lab_b_spec["rho_kgm3"],
        sigma_se=user_consts["sigma_se"],
        eps_pack=user_consts["eps_pack"],
    )
    kwade_vs_nominal_10ms = compute_kwade_stress_scaling(
        d_gm_um=d_gm_pilot_um,
        v_tip_ms=v_tip_pilot_ms,
        phi_gm=phi_gm_pilot,
        rho_gm_kgm3=t_bead_spec["rho_kgm3"],
        SE_opt_mJ=SE_opt_mJ,
        d_gm_lab_um=400.0,
        v_tip_lab_ms=10.0,
        phi_gm_lab=0.80,
        rho_gm_lab_kgm3=6050.0,
        sigma_se=user_consts["sigma_se"],
        eps_pack=user_consts["eps_pack"],
    )

    P_pilot_W = (
        t_model_spec["P_agit_W"]
        * (V_m_pilot / t_model_spec["V_m"])
        * kwade_vs_nominal_10ms["power_density_ratio"]
    )
    pilot_k_params = {
        **base_k_params,
        "S0_ref": base_k_params["S0_ref"] * kwade_vs_lab["S0_multiplier"],
    }

    UA_tank_scaled = UA_t_target * ((V_t_pilot / t_model_spec["V_t_default"]) ** (2.0 / 3.0))

    pilot_selected_cfg = MillLoopConfig(
        V_m=V_m_pilot,
        V_t=V_t_pilot,
        Q=Q_pilot,
        t_max=t_max_pilot,
        rho_slurry=user_consts["rho_slurry"],
        Cp_slurry=user_consts["Cp_slurry"],
        phi_v=cfg.phi_v,
        phi_max=user_consts["phi_max"],
        T_ref=user_consts["T_ref"],
        mu_ref=user_consts["mu_ref"],
        E_mu_over_R=user_consts["E_mu_over_R"],
        P_agit_W=P_pilot_W,
        eta_diss=user_consts["eta_diss"],
        alpha_P=user_consts["alpha_P"],
        UA_mill=UA_m_target,
        UA_tank=UA_tank_scaled,
        T_cool_m=273.15 + T_cool_pilot_C,
        T_cool_t=273.15 + T_cool_pilot_C + 2.0,
        T_init=user_consts["T_ref"],
    )

    t_pilot_grid = np.linspace(0.0, t_max_pilot, 241)
    sim_selected = simulate_coupled_mill_tank_loop(pilot_selected_cfg, pilot_k_params, w0_vec, t_pilot_grid)
    therm_selected = compute_thermal_and_energy_metrics(pilot_selected_cfg, sim_selected)
    cooling_sol = solve_required_pilot_cooling(
        pilot_selected_cfg, pilot_k_params, w0_vec, t_pilot_grid, 273.15 + T_max_limit_C
    )

    chamber_comp_traces = {}
    for ch_name in CHAMBER_LINER_MATERIALS.keys():
        ua_m_ch, ua_t_ch, u_coeff_ch = compute_netzsch_chamber_ua(
            target_model_key,
            ch_name,
            h_slurry_Wm2K=user_consts["h_slurry"],
            h_coolant_Wm2K=user_consts["h_coolant"],
            U_tank_Wm2K=user_consts["U_tank"],
        )
        ch_cfg = MillLoopConfig(
            **{
                **pilot_selected_cfg.__dict__,
                "UA_mill": ua_m_ch,
                "UA_tank": ua_t_ch * ((V_t_pilot / t_model_spec["V_t_default"]) ** (2.0 / 3.0)),
            }
        )
        sim_ch = simulate_coupled_mill_tank_loop(ch_cfg, pilot_k_params, w0_vec, t_pilot_grid)
        chamber_comp_traces[ch_name] = {
            "sim": sim_ch,
            "UA_mill": ua_m_ch,
            "U_coeff": u_coeff_ch,
        }

    def get_batch_endpoint_and_wear(t_arr, d90_arr, E_arr, target_d90, bead_spec, chamber_spec, se_ratio):
        wear_rate_factor = (
            bead_spec["base_wear_ppm_per_kWh_kg"]
            * chamber_spec["wear_factor"]
            * (max(se_ratio, 0.2) ** 0.65)
        )
        wear_traj_ppm = E_arr * wear_rate_factor

        idx = np.where(d90_arr <= target_d90)[0]
        if len(idx) == 0:
            return None, None, float(wear_traj_ppm[-1]), wear_traj_ppm
        i = idx[0]
        if i == 0:
            return float(t_arr[0]), float(E_arr[0]), float(wear_traj_ppm[0]), wear_traj_ppm
        frac = (target_d90 - d90_arr[i - 1]) / (d90_arr[i] - d90_arr[i - 1] - 1e-12)
        t_hit = float(t_arr[i - 1] + frac * (t_arr[i] - t_arr[i - 1]))
        e_hit = float(E_arr[i - 1] + frac * (E_arr[i] - E_arr[i - 1]))
        w_hit = float(wear_traj_ppm[i - 1] + frac * (wear_traj_ppm[i] - wear_traj_ppm[i - 1]))
        return t_hit, e_hit, w_hit, wear_traj_ppm

    t_hit_sel, E_hit_sel, wear_hit_ppm, wear_traj_sel = get_batch_endpoint_and_wear(
        sim_selected["t"],
        sim_selected["d90_t"],
        therm_selected["E_specific_kWh_kg"],
        d90_spec_um,
        t_bead_spec,
        t_chamber_spec,
        kwade_vs_lab["SE_ratio_to_opt"],
    )

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric(
        "Target Agitator Speed",
        f"{pilot_rpm_selected} RPM",
        f"v_t = {v_tip_pilot_ms:.2f} m/s (Ø {d_rotor_target_mm:.0f}mm)",
    )
    k2.metric(
        "Bead Stress Energy SE_GM",
        f"{kwade_vs_lab['SE_pilot_mJ']:.4f} mJ",
        f"{kwade_vs_lab['SE_ratio_to_opt']:.2f}x SE_opt ({t_bead_spec['category']})",
    )
    k3.metric(
        f"Batch Time to {d90_spec_um:.2f} µm",
        f"{t_hit_sel:.1f} min" if t_hit_sel else "> Horizon",
        f"E_m = {E_hit_sel:.3f} kWh/kg" if E_hit_sel else "Increase horizon",
    )
    k4.metric(
        "Peak Mill Temp T_m",
        f"{cooling_sol['peak_T_geom_C']:.1f} °C",
        f"{cooling_sol['peak_T_geom_C'] - T_max_limit_C:+.1f} °C vs Limit",
        delta_color="inverse",
    )
    k5.metric(
        "Shaft Power & Jacket UA",
        f"{P_pilot_W/1000.0:.2f} kW",
        f"UA_m = {UA_m_target:.1f} W/K ({t_chamber_spec['family']})",
    )
    k6.metric(
        f"Est. Media Wear ({t_bead_spec['contaminant_label']})",
        f"{wear_hit_ppm:.1f} ppm",
        "Metal-Free Liner" if t_chamber_spec["metal_free"] else "SS Liner (Fe/Cr/Ni)",
    )

    # Automated 2D Response-Surface Grid Optimizer over (d_GM, RPM / v_t)
    d_grid = np.linspace(150.0, 900.0, 11)
    v_grid = np.linspace(6.0, max_v_tip_allowed, 10)
    rpm_grid = np.array([tip_speed_to_rpm(v, d_rotor_target_mm) for v in v_grid])

    t_batch_surf = np.full((len(v_grid), len(d_grid)), np.nan)
    T_peak_surf = np.zeros((len(v_grid), len(d_grid)))

    ref_time = t_hit_sel if t_hit_sel is not None else t_max_pilot * 1.25
    ref_S0_mult = max(kwade_vs_lab["S0_multiplier"], 1e-4)
    UA_total_avail = UA_m_target + max_supp_UA + UA_tank_scaled * 0.65

    opt_records = []
    for i_v, vt_val in enumerate(v_grid):
        rpm_val = float(rpm_grid[i_v])
        for j_d, dg_val in enumerate(d_grid):
            kw_ij_lab = compute_kwade_stress_scaling(
                dg_val, vt_val, phi_gm_pilot, t_bead_spec["rho_kgm3"], SE_opt_mJ,
                ref_d_gm_lab, ref_v_tip_lab, lab_phi_gm, lab_b_spec["rho_kgm3"],
                sigma_se=user_consts["sigma_se"], eps_pack=user_consts["eps_pack"]
            )
            kw_ij_nom = compute_kwade_stress_scaling(
                dg_val, vt_val, phi_gm_pilot, t_bead_spec["rho_kgm3"], SE_opt_mJ,
                400.0, 10.0, 0.80, 6050.0,
                sigma_se=user_consts["sigma_se"], eps_pack=user_consts["eps_pack"]
            )
            P_ij_W = t_model_spec["P_agit_W"] * (V_m_pilot / t_model_spec["V_m"]) * kw_ij_nom["power_density_ratio"]
            T_peak_ij_C = (
                T_cool_pilot_C
                + (cfg.eta_diss * P_ij_W * 1.12) / max(UA_total_avail, 1.0)
            )
            T_peak_surf[i_v, j_d] = T_peak_ij_C

            t_est_min = ref_time * (ref_S0_mult / max(kw_ij_lab["S0_multiplier"], 1e-4)) ** 0.88
            t_batch_surf[i_v, j_d] = min(t_est_min, t_max_pilot * 1.5)

            solids_kg = (cfg.rho_slurry / 1000.0) * (V_m_pilot + V_t_pilot) * 0.24
            E_est_kWh_kg = (P_ij_W / 1000.0) * (t_est_min / 60.0) / max(solids_kg, 1e-6)
            wear_est_ppm = (
                E_est_kWh_kg
                * t_bead_spec["base_wear_ppm_per_kWh_kg"]
                * t_chamber_spec["wear_factor"]
                * (max(kw_ij_lab["SE_ratio_to_opt"], 0.2) ** 0.65)
            )

            feasible = (T_peak_ij_C <= T_max_limit_C) and (t_est_min <= t_max_pilot)
            opt_records.append({
                "d_GM_um": dg_val,
                "Agitator_RPM": rpm_val,
                "v_tip_ms": vt_val,
                "SE_GM_mJ": kw_ij_lab["SE_pilot_mJ"],
                "eta_SE_pct": kw_ij_lab["eta_pilot"] * 100.0,
                "P_shaft_kW": P_ij_W / 1000.0,
                "T_peak_mitigated_C": T_peak_ij_C,
                "Batch_Time_min": t_est_min,
                "Specific_Energy_kWh_kg": E_est_kWh_kg,
                "Media_Wear_ppm": wear_est_ppm,
                "Thermally_Feasible": feasible,
            })

    df_opt_grid = pd.DataFrame(opt_records)
    df_feas = df_opt_grid[df_opt_grid["Thermally_Feasible"]]
    best_row = (
        df_feas.loc[df_feas["Batch_Time_min"].idxmin()]
        if len(df_feas) > 0
        else df_opt_grid.loc[df_opt_grid["Specific_Energy_kWh_kg"].idxmin()]
    )

    st.info(
        f"🎯 **Kwade Optimum on `{target_model_key[:7]}` ({t_chamber_spec['family']} + {t_bead_spec['category']} Beads):** "
        f"Optimal Bead Diameter **d_GM* = {best_row['d_GM_um']:.0f} µm**, "
        f"Optimal Agitator Speed **N* = {best_row['Agitator_RPM']:.0f} RPM** (**v_t* = {best_row['v_tip_ms']:.2f} m/s**) "
        f"➔ Minimum Batch Time = **{best_row['Batch_Time_min']:.1f} min**, "
        f"Shaft Power = **{best_row['P_shaft_kW']:.2f} kW**, "
        f"Est. Wear = **{best_row['Media_Wear_ppm']:.1f} ppm**."
    )

    y_contour_vals = rpm_grid if contour_y_axis_mode == "Agitator Speed (RPM)" else v_grid
    y_curr_marker = pilot_rpm_selected if contour_y_axis_mode == "Agitator Speed (RPM)" else v_tip_pilot_ms
    y_best_marker = best_row["Agitator_RPM"] if contour_y_axis_mode == "Agitator Speed (RPM)" else best_row["v_tip_ms"]
    y_axis_title = f"Agitator Speed (RPM) [Ø {d_rotor_target_mm:.0f}mm]" if contour_y_axis_mode == "Agitator Speed (RPM)" else "Agitator Tip Speed v_t (m/s)"

    fig_ndv = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"1. Chamber Liner Comparison on {target_model_key[:7]} @ {pilot_rpm_selected} RPM: T_m(t)",
            f"2. Chamber Liner Effect on Holding Tank D90(t) (Target = {d90_spec_um} µm)",
            f"3. Zirconium vs. Ceramic Beads @ {pilot_rpm_selected} RPM: Time & Wear",
            f"4. Kwade 2D Surface: Batch Time (min) vs d_GM & {contour_y_axis_mode}",
        ),
        specs=[[{}, {}], [{"secondary_y": True}, {}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.11,
    )

    ch_colors = {
        "Stainless Steel (AISI 316L Pharma Ra <= 0.4 um)": "#1f77b4",
        "Ceramic - NETZSCH-Ceram C (Sintered SiC)": "#2ca02c",
        "Ceramic - NETZSCH-Ceram N (Silicon Nitride Si3N4)": "#9467bd",
        "Ceramic - NETZSCH-Ceram Z (Zirconia ZrO2)": "#d62728",
    }

    for ch_name, ch_data in chamber_comp_traces.items():
        short_lbl = ch_name.split(" (")[0] + f" (UA={ch_data['UA_mill']:.0f} W/K)"
        is_active = ch_name == target_chamber_key
        fig_ndv.add_trace(
            go.Scatter(
                x=ch_data["sim"]["t"],
                y=ch_data["sim"]["T_m"] - 273.15,
                name=short_lbl,
                line=dict(color=ch_colors[ch_name], width=3.5 if is_active else 2.0, dash="solid" if is_active else "dash"),
            ),
            row=1, col=1,
        )
        fig_ndv.add_trace(
            go.Scatter(
                x=ch_data["sim"]["t"],
                y=ch_data["sim"]["d90_t"],
                name=short_lbl,
                showlegend=False,
                line=dict(color=ch_colors[ch_name], width=3.5 if is_active else 2.0, dash="solid" if is_active else "dash"),
            ),
            row=1, col=2,
        )

    fig_ndv.add_hline(y=T_max_limit_C, line_dash="dot", line_color="red", annotation_text=f"T_max = {T_max_limit_C:.0f} °C", row=1, col=1)
    fig_ndv.add_hline(y=d90_spec_um, line_dash="dot", line_color="black", annotation_text=f"Target D90 = {d90_spec_um} µm", row=1, col=2)

    bead_names_short, bead_t_batch, bead_wear_vals = [], [], []
    for b_name, b_info in BEAD_MEDIA_TYPES.items():
        kw_b = compute_kwade_stress_scaling(
            d_gm_pilot_um, v_tip_pilot_ms, phi_gm_pilot, b_info["rho_kgm3"], SE_opt_mJ,
            ref_d_gm_lab, ref_v_tip_lab, lab_phi_gm, lab_b_spec["rho_kgm3"],
            sigma_se=user_consts["sigma_se"], eps_pack=user_consts["eps_pack"]
        )
        t_est_b = (t_hit_sel if t_hit_sel else t_max_pilot) * (
            max(kwade_vs_lab["S0_multiplier"], 1e-4) / max(kw_b["S0_multiplier"], 1e-4)
        ) ** 0.88
        E_est_b = (E_hit_sel if E_hit_sel else 0.15) * (kw_b["eta_lab"] / max(kw_b["eta_pilot"], 1e-3))
        wear_b = (
            E_est_b
            * b_info["base_wear_ppm_per_kWh_kg"]
            * t_chamber_spec["wear_factor"]
            * (max(kw_b["SE_ratio_to_opt"], 0.2) ** 0.65)
        )
        bead_names_short.append(b_name.split(" (")[0].replace("Zirconium Oxide - ", "Zr: ").replace("Ceramic - ", "Cer: "))
        bead_t_batch.append(float(t_est_b))
        bead_wear_vals.append(float(wear_b))

    fig_ndv.add_trace(
        go.Bar(x=bead_names_short, y=bead_t_batch, name="Predicted Batch Time (min)", marker_color="#1f77b4", opacity=0.85),
        row=2, col=1, secondary_y=False,
    )
    fig_ndv.add_trace(
        go.Scatter(x=bead_names_short, y=bead_wear_vals, name="Predicted Media Wear (ppm)", mode="lines+markers", marker=dict(size=10, color="#d62728"), line=dict(width=3, color="#d62728")),
        row=2, col=1, secondary_y=True,
    )

    fig_ndv.add_trace(
        go.Contour(
            x=d_grid,
            y=y_contour_vals,
            z=t_batch_surf,
            colorscale="Viridis_r",
            showscale=False,
            contours=dict(showlabels=True, labelfont=dict(size=10, color="white")),
            name="Batch Time (min)",
        ),
        row=2, col=2,
    )
    fig_ndv.add_trace(
        go.Contour(
            x=d_grid,
            y=y_contour_vals,
            z=T_peak_surf,
            showscale=False,
            contours=dict(coloring="none", start=T_max_limit_C, end=T_max_limit_C, size=1.0, showlabels=True),
            line=dict(color="#ff2b2b", width=3, dash="dash"),
            name=f"Max Temp Limit ({T_max_limit_C:.0f} °C)",
        ),
        row=2, col=2,
    )
    fig_ndv.add_trace(
        go.Scatter(
            x=[d_gm_pilot_um], y=[y_curr_marker], mode="markers+text", text=["Current"], textposition="top right",
            marker=dict(size=11, color="white", symbol="circle", line=dict(width=3, color="black")),
            name="Selected Setpoint",
        ),
        row=2, col=2,
    )
    fig_ndv.add_trace(
        go.Scatter(
            x=[best_row["d_GM_um"]], y=[y_best_marker], mode="markers+text", text=["Optimum*"], textposition="bottom left",
            marker=dict(size=14, color="#00ff88", symbol="star", line=dict(width=2, color="black")),
            name="Kwade Optimum*",
        ),
        row=2, col=2,
    )

    fig_ndv.update_xaxes(title_text="Batch Time (min)", row=1, col=1)
    fig_ndv.update_yaxes(title_text="Mill Slurry Temp T_m (°C)", row=1, col=1)
    fig_ndv.update_xaxes(title_text="Batch Time (min)", row=1, col=2)
    fig_ndv.update_yaxes(title_text="Holding Tank D90 (µm)", row=1, col=2)
    fig_ndv.update_yaxes(title_text="Batch Time (min)", row=2, col=1, secondary_y=False)
    fig_ndv.update_yaxes(title_text="Wear Contamination (ppm)", row=2, col=1, secondary_y=True)
    fig_ndv.update_xaxes(title_text="Bead Diameter d_GM (µm)", row=2, col=2)
    fig_ndv.update_yaxes(title_text=y_axis_title, row=2, col=2)
    fig_ndv.update_layout(height=780, template="plotly_white", legend=dict(orientation="h", y=-0.14))
    st.plotly_chart(fig_ndv, use_container_width=True)

    st.markdown(
        f"#### 📋 NETZSCH DeltaVita® Fleet Scale-Up Matrix at Matched Tip Speed (**v_t = {v_tip_pilot_ms:.2f} m/s**)"
    )
    fleet_rows = []
    for m_name, m_info in NETZSCH_DELTAVITA_MODELS.items():
        ua_m_ss, ua_t_ss, _ = compute_netzsch_chamber_ua(
            m_name, "Stainless Steel (AISI 316L Pharma Ra <= 0.4 um)",
            user_consts["h_slurry"], user_consts["h_coolant"], user_consts["U_tank"]
        )
        ua_m_sic, _, _ = compute_netzsch_chamber_ua(
            m_name, "Ceramic - NETZSCH-Ceram C (Sintered SiC)",
            user_consts["h_slurry"], user_consts["h_coolant"], user_consts["U_tank"]
        )
        ua_m_zro2, _, _ = compute_netzsch_chamber_ua(
            m_name, "Ceramic - NETZSCH-Ceram Z (Zirconia ZrO2)",
            user_consts["h_slurry"], user_consts["h_coolant"], user_consts["U_tank"]
        )
        ua_m_sel, ua_t_sel, _ = compute_netzsch_chamber_ua(
            m_name, target_chamber_key,
            user_consts["h_slurry"], user_consts["h_coolant"], user_consts["U_tank"]
        )

        rpm_model_matched = tip_speed_to_rpm(v_tip_pilot_ms, m_info["D_rotor_mm"])
        P_m_W = m_info["P_agit_W"] * kwade_vs_nominal_10ms["power_density_ratio"]

        f_cfg = MillLoopConfig(
            V_m=m_info["V_m"],
            V_t=m_info["V_t_default"],
            Q=m_info["Q_default"],
            t_max=t_max_pilot,
            rho_slurry=user_consts["rho_slurry"],
            Cp_slurry=user_consts["Cp_slurry"],
            phi_v=cfg.phi_v,
            phi_max=user_consts["phi_max"],
            T_ref=user_consts["T_ref"],
            mu_ref=user_consts["mu_ref"],
            E_mu_over_R=user_consts["E_mu_over_R"],
            P_agit_W=P_m_W,
            eta_diss=user_consts["eta_diss"],
            alpha_P=user_consts["alpha_P"],
            UA_mill=ua_m_sel,
            UA_tank=ua_t_sel,
            T_cool_m=273.15 + T_cool_pilot_C,
            T_cool_t=273.15 + T_cool_pilot_C + 2.0,
            T_init=user_consts["T_ref"],
        )
        sim_f = simulate_coupled_mill_tank_loop(f_cfg, pilot_k_params, w0_vec, t_pilot_grid)
        therm_f = compute_thermal_and_energy_metrics(f_cfg, sim_f)
        t_f, e_f, w_f, _ = get_batch_endpoint_and_wear(
            sim_f["t"], sim_f["d90_t"], therm_f["E_specific_kWh_kg"],
            d90_spec_um, t_bead_spec, t_chamber_spec, kwade_vs_lab["SE_ratio_to_opt"]
        )

        fleet_rows.append({
            "NETZSCH Model": m_name,
            "V_m (L)": m_info["V_m"],
            "Rotor Ø (mm)": m_info["D_rotor_mm"],
            "Required RPM": rpm_model_matched,
            "Tip Speed (m/s)": v_tip_pilot_ms,
            "Shaft Power P_0 (kW)": P_m_W / 1000.0,
            "UA_m [316L SS] (W/K)": ua_m_ss,
            "UA_m [Ceram C SiC] (W/K)": ua_m_sic,
            "UA_m [Ceram Z ZrO2] (W/K)": ua_m_zro2,
            f"Peak T_m [{t_chamber_spec['family']}] (°C)": float(np.max(sim_f["T_m"]) - 273.15),
            f"Time to {d90_spec_um}µm (min)": f"{t_f:.1f}" if t_f else "> Horizon",
            "Predicted Wear (ppm)": w_f,
        })

    df_fleet = pd.DataFrame(fleet_rows)
    st.dataframe(
        df_fleet.style.format({
            "V_m (L)": "{:.2f}",
            "Rotor Ø (mm)": "{:.0f}",
            "Required RPM": "{:.0f}",
            "Tip Speed (m/s)": "{:.2f}",
            "Shaft Power P_0 (kW)": "{:.2f}",
            "UA_m [316L SS] (W/K)": "{:.1f}",
            "UA_m [Ceram C SiC] (W/K)": "{:.1f}",
            "UA_m [Ceram Z ZrO2] (W/K)": "{:.1f}",
            f"Peak T_m [{t_chamber_spec['family']}] (°C)": "{:.1f}",
            "Predicted Wear (ppm)": "{:.1f}",
        }),
        use_container_width=True,
    )

    col_dl1, col_dl2 = st.columns(2)
    fleet_csv = io.StringIO()
    df_fleet.to_csv(fleet_csv, index=False)
    col_dl1.download_button(
        "📥 Download NETZSCH Fleet RPM & Scale-Up Matrix (CSV)",
        data=fleet_csv.getvalue(),
        file_name="netzsch_deltavita_fleet_rpm_scaleup.csv",
        mime="text/csv",
        use_container_width=True,
    )

    kw_csv = io.StringIO()
    df_opt_grid.to_csv(kw_csv, index=False)
    col_dl2.download_button(
        "📥 Download Kwade (d_GM, RPM, v_t) Response Surface (CSV)",
        data=kw_csv.getvalue(),
        file_name="netzsch_kwade_rpm_stress_energy_surface.csv",
        mime="text/csv",
        use_container_width=True,
    )
