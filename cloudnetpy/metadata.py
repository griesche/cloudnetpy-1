"""Initial Metadata of Cloudnet variables for NetCDF file writing."""

from typing import NamedTuple, Optional


class MetaData(NamedTuple):
    long_name: Optional[str] = None
    standard_name: Optional[str] = None
    units: Optional[str] = None
    comment: Optional[str] = None
    definition: Optional[str] = None
    references: Optional[str] = None
    ancillary_variables: Optional[str] = None
    positive: Optional[str] = None
    axis: Optional[str] = None
    calendar: Optional[str] = None
    source: Optional[str] = None


COMMON_ATTRIBUTES = {
    "time": MetaData(long_name="Time UTC", axis="T", standard_name="time", calendar="standard"),
    "height": MetaData(
        long_name="Height above mean sea level",
        standard_name="height_above_mean_sea_level",
        units="m",
    ),
    "range": MetaData(
        long_name="Range from instrument",
        axis="Z",
        units="m",
        comment="Distance from instrument to centre of each range bin.",
    ),
    "latitude": MetaData(
        long_name="Latitude of site", units="degree_north", standard_name="latitude"
    ),
    "longitude": MetaData(
        long_name="Longitude of site", units="degree_east", standard_name="longitude"
    ),
    "altitude": MetaData(long_name="Altitude of site", standard_name="altitude", units="m"),
    "Zh": MetaData(
        long_name="Radar reflectivity factor",
        units="dBZ",
        comment="Calibrated reflectivity. Calibration convention: in the absence of attenuation,\n"
        "a cloud at 273 K containing one million 100-micron droplets per cubic metre will\n"
        "have a reflectivity of 0 dBZ at all frequencies.",
    ),
    "width": MetaData(
        long_name="Spectral width",
        units="m s-1",
        comment=(
            "This parameter is the standard deviation of the reflectivity-weighted\n"
            "velocities in the radar pulse volume."
        ),
    ),
    "v": MetaData(
        long_name="Doppler velocity",
        units="m s-1",
        comment=(
            "This parameter is the radial component of the velocity, with positive\n"
            "velocities are away from the radar."
        ),
    ),
    "ldr": MetaData(
        long_name="Linear depolarisation ratio",
        units="dB",
        comment="This parameter is the ratio of cross-polar to co-polar reflectivity.",
    ),
    "sldr": MetaData(long_name="Slanted linear depolarisation ratio", units="dB"),
    "lwp": MetaData(
        long_name="Liquid water path",
        units="g m-2",
        standard_name="atmosphere_cloud_liquid_water_content",
    ),
    "iwv": MetaData(
        long_name="Integrated water vapour",
        units="kg m-2",
        standard_name="atmosphere_mass_content_of_water_vapor",
    ),
    "kurtosis": MetaData(
        long_name="Kurtosis of spectra",
        units="1",
    ),
    "nyquist_velocity": MetaData(long_name="Nyquist velocity", units="m s-1"),
    "radar_frequency": MetaData(long_name="Radar transmit frequency", units="GHz"),
    "beta": MetaData(
        long_name="Attenuated backscatter coefficient",
        units="sr-1 m-1",
        comment="SNR-screened attenuated backscatter coefficient.",
    ),
    "beta_raw": MetaData(
        long_name="Attenuated backscatter coefficient",
        units="sr-1 m-1",
        comment="Non-screened attenuated backscatter coefficient.",
    ),
    "beta_smooth": MetaData(
        long_name="Attenuated backscatter coefficient",
        units="sr-1 m-1",
        comment="SNR-screened attenuated backscatter coefficient.\n"
        "Weak background smoothed using Gaussian 2D-kernel.",
    ),
    "wavelength": MetaData(
        long_name="Laser wavelength",
        units="nm",
    ),
    "zenith_angle": MetaData(
        long_name="Zenith angle",
        units="degree",
        standard_name="zenith_angle",
        comment="Angle to the local vertical. A value of zero is directly overhead.",
    ),
    "azimuth_angle": MetaData(
        long_name="Azimuth angle",
        standard_name="sensor_azimuth_angle",
        units="degree",
        comment="Angle between North and the line of sight, measured clockwise.",
    ),
    "temperature": MetaData(
        long_name="Temperature",
        units="K",
    ),
    "pressure": MetaData(
        long_name="Pressure",
        units="Pa",
    ),
    "SNR": MetaData(
        long_name="Signal-to-noise ratio",
        units="dB",
    ),
}
