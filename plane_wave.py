import numpy as np
import cupy as cp
import scipy
from tqdm import tqdm
import multiprocessing as mp
import os
import matplotlib.pyplot as plt
from dataclasses import dataclass
import types
from typing import Any


@dataclass
class MediumParameters:
    x_extent: float
    y_extent: float
    z_extent: float
    num_particles: int

    def _lims(self, wavelength: float, extent: float) -> tuple[float, float]:
        half = wavelength * extent / 2
        return -half, half

    def x_lims(self, wavelength: float) -> tuple[float, float]:
        return self._lims(wavelength, self.x_extent)

    def y_lims(self, wavelength: float) -> tuple[float, float]:
        return self._lims(wavelength, self.y_extent)

    def z_lims(self, wavelength: float) -> tuple[float, float]:
        return self._lims(wavelength, self.z_extent)


@dataclass
class MeasurementPlaneParameters:
    x_extent: float
    y_extent: float
    z: float
    number_x: int
    number_y: int

    def _points(self, wavelength: float, extent: float, xp: Any) -> np.ndarray:
        half = wavelength * extent / 2
        return xp.linspace(-half, half, self.number_x)

    def x_points(self, wavelength: float, xp: Any):
        return self._points(wavelength, self.x_extent, xp)

    def y_points(self, wavelength: float, xp: Any):
        return self._points(wavelength, self.y_extent, xp)

    def _lims(self, wavelength: float, extent: float) -> tuple[float, float]:
        half = wavelength * extent / 2
        return -half, half

    def x_lims(self, wavelength: float) -> tuple[float, float]:
        return self._lims(wavelength, self.x_extent)

    def y_lims(self, wavelength: float) -> tuple[float, float]:
        return self._lims(wavelength, self.y_extent)


@dataclass
class SimulationParameters:
    wavelength: float
    use_cupy: bool
    medium_parameters: MediumParameters
    measurement_plane_parameters: MeasurementPlaneParameters
    seed: int

    @property
    def wavenumber(self) -> float:
        return 2 * np.pi / self.wavelength

    @property
    def xp(self) -> types.ModuleType:
        return cp if self.use_cupy else np

    # -------------------------------------------------------------------------
    # Medium
    # -------------------------------------------------------------------------

    @property
    def medium_x_lims(self) -> tuple[float, float]:
        return self.medium_parameters.x_lims(self.wavelength)

    @property
    def medium_y_lims(self) -> tuple[float, float]:
        return self.medium_parameters.y_lims(self.wavelength)

    @property
    def medium_z_lims(self) -> tuple[float, float]:
        return self.medium_parameters.z_lims(self.wavelength)

    # -------------------------------------------------------------------------
    # Measurement plane
    # -------------------------------------------------------------------------

    @property
    def measurement_x_points(self) -> np.ndarray:
        return self.measurement_plane_parameters.x_points(
            self.wavelength, self.xp
        )

    @property
    def measurement_y_points(self) -> np.ndarray:
        return self.measurement_plane_parameters.y_points(
            self.wavelength, self.xp
        )

    @property
    def measurement_z(self) -> np.ndarray:
        return self.measurement_plane_parameters.z * self.wavelength

    def print_medium_limits(self) -> None:
        print(f"The scattering medium extends from:")
        print(
            f"{self.medium_x_lims[0]*1e6:.1f} um < x < {self.medium_x_lims[1]*1e6:.1f} um"
        )
        print(
            f"{self.medium_y_lims[0]*1e6:.1f} um < y < {self.medium_y_lims[1]*1e6:.1f} um"
        )
        print(
            f"{self.medium_z_lims[0]*1e6:.1f} um < z < {self.medium_z_lims[1]*1e6:.1f} um"
        )


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def plane_wave(
    x: np.ndarray | cp.ndarray,
    y: np.ndarray | cp.ndarray,
    z: np.ndarray | cp.ndarray,
    k: np.ndarray | cp.ndarray,
    sp: SimulationParameters,
) -> np.ndarray:
    """E = exp(ik.r)"""
    phase = 1j * sp.wavenumber * (k[0] * x + k[1] * y + k[2] * z)
    return np.exp(phase)


def plane_wave_vector(
    r: np.ndarray | cp.ndarray,
    k: np.ndarray | cp.ndarray,
    pol: np.ndarray | cp.ndarray,
    sp: SimulationParameters,
):
    """Electric field associated with a plane wave for a given wavevector and
    polarization."""
    xp = sp.xp
    k = k[:, :, None]
    pol = pol[:, :, None]
    phase = sp.wavenumber * xp.sum(k * r[None, :, :], axis=1)
    E = pol * xp.exp(1j * phase[:, None, :])
    return E


def get_particle_locations(sp: SimulationParameters, seed: bool = True):
    """Generate uniformly random particle positions within given bounds."""
    xp = sp.xp

    if seed:
        seed = sp.seed
        xp.random.seed(seed)

    x = xp.random.uniform(
        sp.medium_x_lims[0],
        sp.medium_x_lims[1],
        size=sp.medium_parameters.num_particles,
    )
    y = xp.random.uniform(
        sp.medium_y_lims[0],
        sp.medium_y_lims[1],
        size=sp.medium_parameters.num_particles,
    )
    z = xp.random.uniform(
        sp.medium_z_lims[0],
        sp.medium_z_lims[1],
        size=sp.medium_parameters.num_particles,
    )
    return x, y, z


def get_incident_field(
    incident_wavevector: np.ndarray | cp.ndarray,
    sp: SimulationParameters,
) -> np.ndarray | cp.ndarray:
    xp = sp.xp
    k = sp.wavenumber

    xm = sp.measurement_x_points
    ym = sp.measurement_y_points
    Xm, Ym = xp.meshgrid(xm, ym, indexing="xy")
    Zm = sp.measurement_z * xp.ones_like(Xm)

    field = plane_wave(Xm, Ym, Zm, incident_wavevector, sp)
    return field


def get_scattered_field(
    x_particles: np.ndarray | cp.ndarray,
    y_particles: np.ndarray | cp.ndarray,
    z_particles: np.ndarray | cp.ndarray,
    incident_wavevector: np.ndarray | cp.ndarray,
    sp: SimulationParameters,
) -> np.ndarray | cp.ndarray:
    xp = sp.xp
    k = sp.wavenumber

    xm = sp.measurement_x_points
    ym = sp.measurement_y_points
    Xm, Ym = xp.meshgrid(xm, ym, indexing="xy")
    Zm = sp.measurement_z * xp.ones_like(Xm)

    field = xp.zeros_like(Xm, dtype=complex)
    # Loop over each particle. Total field is the sum
    for x_i, y_i, z_i in zip(x_particles, y_particles, z_particles):
        # Calculate the incident field at the particle
        E_i = plane_wave(x_i, y_i, z_i, incident_wavevector, sp)

        r = xp.sqrt((Xm - x_i) ** 2 + (Ym - y_i) ** 2 + (Zm - z_i) ** 2)
        field += E_i * xp.exp(1j * k * r) / (k * r)

    return field
