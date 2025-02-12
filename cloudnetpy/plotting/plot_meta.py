"""Metadata for plotting module."""
from enum import Enum
from typing import NamedTuple, Optional, Sequence, Tuple, Union


class Scale(Enum):
    LOGARITHMIC = "logarithmic"
    LINEAR = "linear"


class PlotMeta(NamedTuple):
    name: str
    cbar: Optional[Union[str, Sequence[str]]] = None
    clabel: Optional[Union[str, Sequence[Tuple[str, str]]]] = None
    ylabel: Optional[str] = None
    plot_range: Optional[Tuple[float, float]] = None
    plot_scale: Optional[Scale] = None
    plot_type: Optional[str] = None
    source: Optional[str] = None


_MUM = "$\\mu$m"
_M3 = "$m^{-3}$"
_MS1 = "m s$^{-1}$"
_SR1M1 = "sr$^{-1}$ m$^{-1}$"
_KGM2 = "kg m$^{-2}$"
_KGM3 = "kg m$^{-3}$"
_KGM2S1 = "kg m$^{-2}$ s$^{-1}$"
_DB = "dB"
_DBZ = "dBZ"

_COLORS = {
    "green": "#3cb371",
    "darkgreen": "#253A24",
    "lightgreen": "#70EB5D",
    "yellowgreen": "#C7FA3A",
    "yellow": "#FFE744",
    "orange": "#ffa500",
    "pink": "#B43757",
    "red": "#F57150",
    "shockred": "#E64A23",
    "seaweed": "#646F5E",
    "seaweed_roll": "#748269",
    "white": "#ffffff",
    "lightblue": "#6CFFEC",
    "blue": "#209FF3",
    "skyblue": "#CDF5F6",
    "darksky": "#76A9AB",
    "darkpurple": "#464AB9",
    "lightpurple": "#6A5ACD",
    "purple": "#BF9AFF",
    "darkgray": "#2f4f4f",
    "lightgray": "#ECECEC",
    "gray": "#d3d3d3",
    "lightbrown": "#CEBC89",
    "lightsteel": "#a0b0bb",
    "steelblue": "#4682b4",
    "mask": "#C8C8C8",
}

# Labels (and corresponding data) starting with an underscore are NOT shown:

_CLABEL = {
    "target_classification": (
        ("_Clear sky", _COLORS["white"]),
        ("Droplets", _COLORS["lightblue"]),
        ("Drizzle or rain", _COLORS["blue"]),
        ("Drizzle & droplets", _COLORS["purple"]),
        ("Ice", _COLORS["lightsteel"]),
        ("Ice & droplets", _COLORS["darkpurple"]),
        ("Melting ice", _COLORS["orange"]),
        ("Melting & droplets", _COLORS["yellowgreen"]),
        ("Aerosols", _COLORS["lightbrown"]),
        ("Insects", _COLORS["shockred"]),
        ("Aerosols & insects", _COLORS["pink"]),
        ("No data", _COLORS["mask"]),
    ),
    "detection_status": (
        ("_Clear sky", _COLORS["white"]),
        ("Lidar only", _COLORS["yellow"]),
        ("Uncorrected atten.", _COLORS["seaweed_roll"]),
        ("Radar & lidar", _COLORS["green"]),
        ("_No radar but unknown atten.", _COLORS["purple"]),
        ("Radar only", _COLORS["lightgreen"]),
        ("_No radar but known atten.", _COLORS["orange"]),
        ("Corrected atten.", _COLORS["skyblue"]),
        ("Clutter", _COLORS["shockred"]),
        ("_Lidar molecular scattering", _COLORS["pink"]),
        ("No data", _COLORS["mask"]),
    ),
    "ice_retrieval_status": (
        ("_No ice", _COLORS["white"]),
        ("Reliable", _COLORS["green"]),
        ("Uncorrected", _COLORS["orange"]),
        ("Corrected", _COLORS["lightgreen"]),
        ("Ice from lidar", _COLORS["yellow"]),
        ("Ice above rain", _COLORS["darksky"]),
        ("Clear above rain", _COLORS["skyblue"]),
        ("Positive temp.", _COLORS["seaweed"]),
        ("No data", _COLORS["mask"]),
    ),
    "lwc_retrieval_status": (
        ("No liquid", _COLORS["white"]),
        ("Reliable", _COLORS["green"]),
        ("Adjusted", _COLORS["lightgreen"]),
        ("New pixel", _COLORS["yellow"]),
        ("Invalid LWP", _COLORS["seaweed_roll"]),
        ("_Invalid LWP2", _COLORS["shockred"]),
        ("_Measured rain", _COLORS["orange"]),
        ("No data", _COLORS["mask"]),
    ),
    "drizzle_retrieval_status": (
        ("_No drizzle", _COLORS["white"]),
        ("Reliable", _COLORS["green"]),
        ("Below melting", _COLORS["lightgreen"]),
        ("Unfeasible", _COLORS["red"]),
        ("Drizzle-free", _COLORS["orange"]),
        ("Rain", _COLORS["seaweed"]),
        ("No data", _COLORS["mask"]),
    ),
    "der_retrieval_status": (
        ("_Clear sky", _COLORS["white"]),
        ("Reliable", _COLORS["green"]),
        ("Mixed-phase", _COLORS["lightgreen"]),
        ("Unfeasible", _COLORS["red"]),
        ("Surrounding-ice", _COLORS["mask"]),
    ),
}

_CBAR = {"bit": (_COLORS["white"], _COLORS["steelblue"])}

ATTRIBUTES = {
    "ier": PlotMeta(
        name="Ice effective radius",
        cbar="viridis",
        clabel=_MUM,
        plot_range=(20, 60),
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "ier_inc_rain": PlotMeta(
        name="Ice effective radius (including rain)",
        cbar="Blues",
        clabel=_MUM,
        plot_range=(20, 60),
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "ier_error": PlotMeta(
        name="Ice effective radius error",
        cbar="RdYlGn_r",
        clabel=_MUM,
        plot_range=(10, 50),
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "ier_retrieval_status": PlotMeta(
        name="Ice effective radius retrieval status",
        clabel=_CLABEL["ice_retrieval_status"],
        plot_type="segment",
    ),
    "Do": PlotMeta(
        name="Drizzle median diameter",
        cbar="viridis",
        clabel="m",
        plot_range=(1e-6, 1e-3),
        plot_scale=Scale.LOGARITHMIC,
        plot_type="mesh",
    ),
    "Do_error": PlotMeta(
        name="Random error in drizzle median diameter",
        cbar="RdYlGn_r",
        clabel=_DB,
        plot_range=(0.1, 0.5),
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "der": PlotMeta(
        name="Droplet effective radius",
        cbar="coolwarm",
        clabel="m",
        plot_range=(1.0e-6, 1.0e-4),
        plot_scale=Scale.LOGARITHMIC,
        plot_type="mesh",
    ),
    "N_scaled": PlotMeta(
        name="Cloud droplet number concentration",
        cbar="viridis",
        clabel="",
        plot_range=(1.0e0, 1e3),
        plot_scale=Scale.LOGARITHMIC,
        plot_type="mesh",
    ),
    "der_error": PlotMeta(
        name="Absolute error in effective radius",
        cbar="coolwarm",
        clabel="m",
        plot_range=(1.0e-6, 1.0e-4),
        plot_scale=Scale.LOGARITHMIC,
        plot_type="mesh",
    ),
    "der_scaled": PlotMeta(
        name="Droplet effective radius (scaled to LWP)",
        cbar="coolwarm",
        clabel="m",
        plot_range=(1.0e-6, 1.0e-4),
        plot_scale=Scale.LOGARITHMIC,
        plot_type="mesh",
    ),
    "der_scaled_error": PlotMeta(
        name="Absolute error in effective radius (scaled to LWP)",
        cbar="coolwarm",
        clabel="m",
        plot_range=(1.0e-6, 1.0e-4),
        plot_scale=Scale.LOGARITHMIC,
        plot_type="mesh",
    ),
    "der_retrieval_status": PlotMeta(
        name="Effective radius retrieval status",
        clabel=_CLABEL["der_retrieval_status"],
        plot_type="segment",
    ),
    "mu": PlotMeta(
        name="Drizzle droplet size distribution shape parameter",
        cbar="viridis",
        clabel="",
        plot_range=(0, 10),
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "S": PlotMeta(
        name="Backscatter-to-extinction ratio",
        cbar="viridis",
        clabel="",
        plot_range=(0, 25),
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "S_error": PlotMeta(
        name="Random error in backscatter-to-extinction ratio",
        cbar="RdYlGn_r",
        clabel=_DB,
        plot_range=(0.1, 0.5),
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "drizzle_N": PlotMeta(
        name="Drizzle number concentration",
        cbar="viridis",
        clabel=_M3,
        plot_range=(1e4, 1e9),
        plot_scale=Scale.LOGARITHMIC,
        plot_type="mesh",
    ),
    "drizzle_N_error": PlotMeta(
        name="Random error in drizzle number concentration",
        cbar="RdYlGn_r",
        clabel=_DB,
        plot_range=(0.1, 0.5),
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "drizzle_lwc": PlotMeta(
        name="Drizzle liquid water content",
        cbar="viridis",
        clabel=_KGM3,
        plot_range=(1e-8, 1e-3),
        plot_scale=Scale.LOGARITHMIC,
        plot_type="mesh",
    ),
    "drizzle_lwc_error": PlotMeta(
        name="Random error in drizzle liquid water content",
        cbar="RdYlGn_r",
        clabel=_DB,
        plot_range=(0.3, 1),
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "drizzle_lwf": PlotMeta(
        name="Drizzle liquid water flux",
        cbar="viridis",
        clabel=_KGM2S1,
        plot_range=(1e-8, 1e-5),
        plot_scale=Scale.LOGARITHMIC,
        plot_type="mesh",
    ),
    "drizzle_lwf_error": PlotMeta(
        name="Random error in drizzle liquid water flux",
        cbar="RdYlGn_r",
        clabel=_DB,
        plot_range=(0.3, 1),
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "v_drizzle": PlotMeta(
        name="Drizzle droplet fall velocity",
        cbar="RdBu_r",
        clabel=_MS1,
        plot_range=(-2, 2),
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "v_drizzle_error": PlotMeta(
        name="Random error in drizzle droplet fall velocity",
        cbar="RdYlGn_r",
        clabel=_DB,
        plot_range=(0.3, 1),
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "drizzle_retrieval_status": PlotMeta(
        name="Drizzle parameter retrieval status",
        clabel=_CLABEL["drizzle_retrieval_status"],
        plot_type="segment",
    ),
    "v_air": PlotMeta(
        name="Vertical air velocity",
        cbar="RdBu_r",
        clabel=_MS1,
        plot_range=(-2, 2),
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "uwind": PlotMeta(
        name="Model zonal wind",
        cbar="RdBu_r",
        clabel=_MS1,
        plot_range=(-50, 50),
        plot_scale=Scale.LINEAR,
        plot_type="model",
    ),
    "vwind": PlotMeta(
        name="Model meridional wind",
        cbar="RdBu_r",
        clabel=_MS1,
        plot_range=(-50, 50),
        plot_scale=Scale.LINEAR,
        plot_type="model",
    ),
    "temperature": PlotMeta(
        name="Model temperature",
        cbar="RdBu_r",
        clabel="K",
        plot_range=(223.15, 323.15),
        plot_scale=Scale.LINEAR,
        plot_type="model",
    ),
    "cloud_fraction": PlotMeta(
        name="Cloud fraction",
        cbar="Blues",
        clabel="",
        plot_range=(0, 1),
        plot_scale=Scale.LINEAR,
        plot_type="model",
    ),
    "Tw": PlotMeta(
        name="Wet-bulb temperature",
        cbar="RdBu_r",
        clabel="K",
        plot_range=(223.15, 323.15),
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "specific_humidity": PlotMeta(
        name="Model specific humidity",
        cbar="viridis",
        clabel="",
        plot_range=(1e-5, 1e-2),
        plot_scale=Scale.LOGARITHMIC,
        plot_type="model",
    ),
    "q": PlotMeta(
        name="Model specific humidity",
        cbar="viridis",
        clabel="",
        plot_range=(1e-5, 1e-2),
        plot_scale=Scale.LOGARITHMIC,
        plot_type="model",
    ),
    "pressure": PlotMeta(
        name="Model pressure",
        cbar="viridis",
        clabel="Pa",
        plot_range=(1e4, 1.5e5),
        plot_scale=Scale.LOGARITHMIC,
        plot_type="model",
    ),
    "beta": PlotMeta(
        name="Attenuated backscatter coefficient",
        cbar="viridis",
        clabel=_SR1M1,
        plot_range=(1e-7, 1e-4),
        plot_scale=Scale.LOGARITHMIC,
        plot_type="mesh",
    ),
    "beta_raw": PlotMeta(
        name="Raw attenuated backscatter coefficient",
        cbar="viridis",
        clabel=_SR1M1,
        plot_range=(1e-7, 1e-4),
        plot_scale=Scale.LOGARITHMIC,
        plot_type="mesh",
    ),
    "beta_smooth": PlotMeta(
        name="Attenuated backscatter coefficient (smoothed)",
        cbar="viridis",
        clabel=_SR1M1,
        plot_range=(1e-7, 1e-4),
        plot_scale=Scale.LOGARITHMIC,
        plot_type="mesh",
    ),
    "depolarisation_raw": PlotMeta(
        name="Raw depolarisation",
        cbar="viridis",
        clabel="",
        plot_range=(1e-3, 1),
        plot_scale=Scale.LOGARITHMIC,
        plot_type="mesh",
    ),
    "depolarisation": PlotMeta(
        name="Lidar depolarisation",
        cbar="viridis",
        clabel="",
        plot_range=(1e-3, 1),
        plot_scale=Scale.LOGARITHMIC,
        plot_type="mesh",
    ),
    "depolarisation_smooth": PlotMeta(
        name="Lidar depolarisation (smoothed)",
        cbar="viridis",
        clabel="",
        plot_range=(1e-3, 1),
        plot_scale=Scale.LOGARITHMIC,
        plot_type="mesh",
    ),
    "Z": PlotMeta(
        name="Radar reflectivity factor",
        cbar="viridis",
        clabel=_DBZ,
        plot_range=(-40, 15),
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "Z_error": PlotMeta(
        name="Radar reflectivity factor random error",
        cbar="RdYlGn_r",
        clabel=_DB,
        plot_range=(0, 3),
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "Zh": PlotMeta(
        name="Radar reflectivity factor",
        cbar="viridis",
        clabel=_DBZ,
        plot_range=(-40, 15),
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "ldr": PlotMeta(
        name="Linear depolarisation ratio",
        cbar="viridis",
        clabel=_DB,
        plot_range=(-30, -5),
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "sldr": PlotMeta(
        name="Slanted linear depolarisation ratio",
        cbar="viridis",
        clabel=_DB,
        plot_range=(-30, -5),
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "zdr": PlotMeta(
        name="Differential reflectivity",
        cbar="RdBu_r",
        clabel=_DB,
        plot_range=(-1, 1),
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "width": PlotMeta(
        name="Spectral width",
        cbar="viridis",
        clabel=_MS1,
        plot_range=(1e-2, 1e0),
        plot_scale=Scale.LOGARITHMIC,
        plot_type="mesh",
    ),
    "v": PlotMeta(
        name="Doppler velocity",
        cbar="RdBu_r",
        clabel=_MS1,
        plot_range=(-4, 4),
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "skewness": PlotMeta(
        name="Skewness",
        cbar="RdBu_r",
        clabel="",
        plot_range=(-1, 1),
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "kurtosis": PlotMeta(
        name="Kurtosis",
        cbar="viridis",
        clabel="",
        plot_range=(1, 5),
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "phi_cx": PlotMeta(
        name="Co-cross-channel differential phase",
        cbar="RdBu_r",
        clabel="rad",
        plot_range=(-2, 2),
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "differential_attenuation": PlotMeta(
        name="Differential attenuation",
        cbar="viridis",
        clabel="dB km-1",
        plot_range=(0, 1),  # TODO: Check
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "rho_cx": PlotMeta(
        name="Co-cross-channel correlation coefficient",
        cbar="viridis",
        clabel="",
        plot_range=(1e-2, 1e0),
        plot_scale=Scale.LOGARITHMIC,
        plot_type="mesh",
    ),
    "rho_hv": PlotMeta(
        name="Correlation coefficient",
        cbar="viridis",
        clabel="",
        plot_range=(0.8, 1),
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "srho_hv": PlotMeta(
        name="Slanted correlation coefficient",
        cbar="viridis",
        clabel="",
        plot_range=(0, 0.5),
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "phi_dp": PlotMeta(
        name="Differential phase",
        cbar="RdBu_r",
        clabel="rad",
        plot_range=(-0.1, 0.1),
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "kdp": PlotMeta(
        name="Specific differential phase shift",
        cbar="RdBu_r",
        clabel="rad km-1",
        plot_range=(-0.1, 0.1),
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "v_sigma": PlotMeta(
        name="Standard deviation of mean velocity",
        cbar="viridis",
        clabel=_MS1,
        plot_range=(1e-2, 1e0),
        plot_scale=Scale.LOGARITHMIC,
        plot_type="mesh",
    ),
    "insect_prob": PlotMeta(
        name="Insect probability",
        cbar="viridis",
        clabel="",
        plot_range=(0, 1),
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "radar_liquid_atten": PlotMeta(
        name="Approximate two-way radar attenuation due to liquid water",
        cbar="viridis",
        clabel=_DB,
        plot_range=(0, 5),
        plot_scale=Scale.LINEAR,  # already logarithmic
        plot_type="mesh",
    ),
    "radar_gas_atten": PlotMeta(
        name="Two-way radar attenuation due to atmospheric gases",
        cbar="viridis",
        clabel=_DB,
        plot_range=(0, 1),
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "lwp": PlotMeta(
        name="Liquid water path",
        cbar="Blues",
        ylabel=_KGM2,
        plot_range=(0, 1),
        plot_scale=Scale.LINEAR,
        plot_type="bar",
        source="mwr",
    ),
    "iwv": PlotMeta(
        name="Integrated water vapour",
        cbar="Blues",
        ylabel=_KGM2,
        plot_range=(0, 1),
        plot_scale=Scale.LINEAR,
        plot_type="bar",
        source="mwr",
    ),
    "rainfall_rate": PlotMeta(name="Rainfall rate", plot_type="bar", source="disdrometer"),
    "n_particles": PlotMeta(name="Number of particles", plot_type="bar", source="disdrometer"),
    "target_classification": PlotMeta(
        name="Target classification", clabel=_CLABEL["target_classification"], plot_type="segment"
    ),
    "detection_status": PlotMeta(
        name="Radar and lidar detection status",
        clabel=_CLABEL["detection_status"],
        plot_type="segment",
    ),
    "iwc": PlotMeta(
        name="Ice water content",
        cbar="viridis",
        clabel=_KGM3,
        plot_range=(1e-7, 1e-3),
        plot_scale=Scale.LOGARITHMIC,
        plot_type="mesh",
    ),
    "iwc_inc_rain": PlotMeta(
        name="Ice water content (including rain)",
        cbar="Blues",
        clabel=_KGM3,
        plot_range=(1e-7, 1e-4),
        plot_scale=Scale.LOGARITHMIC,
        plot_type="mesh",
    ),
    "iwc_error": PlotMeta(
        name="Ice water content error",
        cbar="RdYlGn_r",
        clabel=_DB,
        plot_range=(0, 5),
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "iwc_retrieval_status": PlotMeta(
        name="Ice water content retrieval status",
        clabel=_CLABEL["ice_retrieval_status"],
        plot_type="segment",
    ),
    "lwc": PlotMeta(
        name="Liquid water content",
        cbar="Blues",
        clabel=_KGM3,
        plot_range=(1e-5, 1e-2),
        plot_scale=Scale.LOGARITHMIC,
        plot_type="mesh",
    ),
    "lwc_error": PlotMeta(
        name="Liquid water content error",
        cbar="RdYlGn_r",
        clabel=_DB,
        plot_range=(0, 2),
        plot_scale=Scale.LINEAR,
        plot_type="mesh",
    ),
    "lwc_retrieval_status": PlotMeta(
        name="Liquid water content retrieval status",
        clabel=_CLABEL["lwc_retrieval_status"],
        plot_type="segment",
    ),
    "droplet": PlotMeta(name="Droplet bit", cbar=_CBAR["bit"], plot_range=(0, 1), plot_type="bit"),
    "falling": PlotMeta(name="Falling bit", cbar=_CBAR["bit"], plot_range=(0, 1), plot_type="bit"),
    "cold": PlotMeta(name="Cold bit", cbar=_CBAR["bit"], plot_range=(0, 1), plot_type="bit"),
    "melting": PlotMeta(name="Melting bit", cbar=_CBAR["bit"], plot_range=(0, 1), plot_type="bit"),
    "aerosol": PlotMeta(name="Aerosol bit", cbar=_CBAR["bit"], plot_range=(0, 1), plot_type="bit"),
    "insect": PlotMeta(name="Insect bit", cbar=_CBAR["bit"], plot_range=(0, 1), plot_type="bit"),
    "radar": PlotMeta(name="Radar bit", cbar=_CBAR["bit"], plot_range=(0, 1), plot_type="bit"),
    "lidar": PlotMeta(name="Lidar bit", cbar=_CBAR["bit"], plot_range=(0, 1), plot_type="bit"),
    "clutter": PlotMeta(name="Clutter bit", cbar=_CBAR["bit"], plot_range=(0, 1), plot_type="bit"),
    "molecular": PlotMeta(
        name="Molecular bit", cbar=_CBAR["bit"], plot_range=(0, 1), plot_type="bit"
    ),
    "attenuated": PlotMeta(
        name="Attenuated bit", cbar=_CBAR["bit"], plot_range=(0, 1), plot_type="bit"
    ),
    "corrected": PlotMeta(
        name="Corrected bit", cbar=_CBAR["bit"], plot_range=(0, 1), plot_type="bit"
    ),
}
