import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from constants import *


def safe_normalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Safely normalizes an array of vectors along the specified axis,
    preventing division by zero.
    """
    norms = np.linalg.norm(v, axis=axis, keepdims=True)
    return np.where(norms == 0, 0.0, v / norms)


class DotDict(dict):
    """
    A clean dictionary subclass that supports dot-notation attribute access.
    Unifies attribute access and dictionary keys into a single storage mechanism.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, AttributeError):
            return default


class SchemaDict(DotDict):
    """
    A DotDict that ensures default attributes are always initialized
    while seamlessly accepting any extra user-defined keys.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ComponentState(SchemaDict):
    """
    Base class for sub-states that represent collections of assets/components
    stored as parallel NumPy arrays and lists.

    Provides standard utility methods:
    - len(state): returns length based on NumPy array or list dimensions.
    - state.subset(mask): returns a sliced copy of the component state.
    - state.append(other_state): appends another component state in-place.
    """

    def __len__(self) -> int:
        for val in self.values():
            if isinstance(val, np.ndarray):
                return val.shape[0] if val.ndim >= 1 else 0
            elif isinstance(val, list):
                return len(val)
        return 0

    def subset(self, mask: Any) -> "ComponentState":
        """
        Slices all 1D/2D NumPy array fields and list fields by the given mask
        and returns a new instance of the same class.
        """
        cls = self.__class__
        new_kwargs = {}
        n_total = len(self)

        for key, val in self.items():
            if isinstance(val, np.ndarray):
                if val.ndim == 1:
                    new_kwargs[key] = val[mask]
                elif val.ndim == 2:
                    if val.shape[0] == n_total:
                        new_kwargs[key] = val[mask, :]
                    elif val.shape[1] == n_total:
                        new_kwargs[key] = val[:, mask]
                    else:
                        new_kwargs[key] = val[mask]
                else:
                    new_kwargs[key] = val[mask]
            elif isinstance(val, list):
                if isinstance(mask, np.ndarray) and mask.dtype == bool:
                    indices = np.where(mask)[0]
                elif isinstance(mask, slice):
                    indices = range(*mask.indices(len(val)))
                else:
                    indices = mask
                new_kwargs[key] = [val[i] for i in indices]
            else:
                new_kwargs[key] = val

        return cls(**new_kwargs)

    def append(self, other: "ComponentState") -> None:
        """
        Appends all matching NumPy arrays and list attributes from `other` in-place.
        """
        for key, val in other.items():
            if key in self:
                target = self[key]
                if isinstance(target, np.ndarray) and isinstance(val, np.ndarray):
                    if target.ndim == 1:
                        self[key] = np.append(target, val)
                    elif target.ndim == 2:
                        if target.shape[1] == val.shape[1]:
                            self[key] = np.vstack([target, val])
                        elif target.shape[0] == val.shape[0]:
                            self[key] = np.hstack([target, val])
                        else:
                            self[key] = np.vstack([target, val])
                elif isinstance(target, list) and isinstance(val, list):
                    target.extend(val)


class CountsState(SchemaDict):
    """
    CountsState provides access to asset counts.
    If sim_data reference is available, it dynamically returns
    the actual length of component arrays.
    """
    def __init__(self, parent_sim=None, **kwargs):
        defaults = {
            'celestial': 0,
            'satellites': 0,
            'observatories': 0,
            'fixedpoints': 0,
        }
        dict.__setattr__(self, '_parent_sim', parent_sim)
        super().__init__(**{**defaults, **kwargs})

    def __getitem__(self, item):
        if hasattr(self, '_parent_sim') and self._parent_sim is not None:
            comp = self._parent_sim.get(item)
            if comp is not None and hasattr(comp, '__len__'):
                return len(comp)
        return super().__getitem__(item)

    def __getattr__(self, name):
        if name.startswith('_'):
            return super().__getattr__(name)
        if hasattr(self, '_parent_sim') and self._parent_sim is not None:
            comp = self._parent_sim.get(name)
            if comp is not None and hasattr(comp, '__len__'):
                return len(comp)
        return super().__getattr__(name)

    def __contains__(self, item):
        if super().__contains__(item):
            return True
        if (dict.__getattribute__(self, '_parent_sim') is not None):
            comp = self._parent_sim.get(item)
            if comp is not None and hasattr(comp, '__len__'):
                return True
        return False

    def items(self):
        return [(k, self[k]) for k in self.keys() if not k.startswith('_')]

    def values(self):
        return [self[k] for k in self.keys() if not k.startswith('_')]


class CelestialState(ComponentState):
    def __init__(self, **kwargs):
        defaults = {
            'position': np.zeros((2, 3)),
            'velocity': np.zeros((2, 3)),
            'acceleration': np.zeros((2, 3)),
        }
        super().__init__(**{**defaults, **kwargs})


class SatellitesState(ComponentState):
    def __init__(self, **kwargs):
        defaults = {
            'position': np.zeros((0, 3)),
            'velocity': np.zeros((0, 3)),
            'acceleration': np.zeros((0, 3)),
            'orbital_elements': np.zeros((0, 6)),
            'epochs': [],
            'detector': None,
        }
        super().__init__(**{**defaults, **kwargs})


class FixedPointsState(ComponentState):
    def __init__(self, **kwargs):
        defaults = {
            'position': np.zeros((0, 3)),
            'exclusion': np.zeros((0,)),
            'size': np.zeros((0,)),
            'albedo': np.zeros((0,)),
        }
        super().__init__(**{**defaults, **kwargs})

    def add_target(self, position: np.ndarray, size: float, albedo: float = 0.2) -> None:
        pos = np.asarray(position, dtype=float).reshape(-1, 3)
        n_new = pos.shape[0]
        self.position = np.vstack([self.position, pos]) if self.position.size else pos
        self.exclusion = np.append(self.exclusion, np.zeros(n_new, dtype=int))
        self.size = np.append(self.size, np.full(n_new, size, dtype=float))
        self.albedo = np.append(self.albedo, np.full(n_new, albedo, dtype=float))


class ObservatoriesState(ComponentState):
    def __init__(self, **kwargs):
        defaults = {
            'position': np.zeros((0, 3)),
            'velocity': np.zeros((0, 3)),
            'acceleration': np.zeros((0, 3)),
            'pointing': np.zeros((0, 3)),
            'latitude': np.zeros(0),
            'longitude': np.zeros(0),
            'altitude': np.zeros(0),
            'detector': None,
        }
        super().__init__(**{**defaults, **kwargs})


@dataclass
class CadenceGroup:
    """
    Holds scheduling information for one group of detectors that share
    an identical integration time.

    Attributes:
        scanInterval: Integration period in seconds for this group.
        scanMask:     Boolean array (length = total detectors) selecting
                      the detectors that belong to this group.
        scanNext:     The datetime at which the next scan for this group
                      is due.  Initialised to sim_data.time so that every
                      group fires on the first call to nextIntegration.
    """
    scanInterval: float
    scanMask: np.ndarray
    scanNext: datetime


class CadenceState(SchemaDict):
    """
    Top-level cadence schedule stored in sim_data.cadenceStructure.

    Attributes:
        cadenceList: Ordered list of CadenceGroup objects, one per unique
                     integration time found in sim_data.detector.
        nextTime:    The datetime of the earliest upcoming scan across all
                     groups.
        nextGroup:   Index into cadenceList identifying which group fires
                     next.
    """
    def __init__(self, **kwargs):
        defaults = {
            'cadenceList': [],
            'nextTime': None,
            'nextGroup': 0,
        }
        super().__init__(**{**defaults, **kwargs})


class SimulationState(SchemaDict):
    """
    SimulationState models the main simulation data structure.
    It inherits from SchemaDict to maintain 100% backward compatibility
    with dict-style lookup/subscription and isinstance(..., dict) checks,
    but supports attribute-style access for better structure.
    """
    _FIELD_SCHEMAS = {
        'counts': CountsState,
        'celestial': CelestialState,
        'satellites': SatellitesState,
        'fixedpoints': FixedPointsState,
        'observatories': ObservatoriesState,
        'cadenceStructure': CadenceState,
    }

    def __init__(self, **kwargs):
        if 'start_time' not in kwargs:
            raise TypeError("SimulationState requires a 'start_time' parameter.")
        start = kwargs['start_time']

        defaults = {
            'time': start,
            'delta_time': 60.0,
            'counts': None,
            'pointing_spheres': {},
            'satellites': None,
            'observatories': None,
            'fixedpoints': None,
            'celestial': None,
            'cadenceStructure': None,
        }
        super().__init__(**{**defaults, **kwargs})
        self.counts = CountsState(parent_sim=self)

    @property
    def detector(self):
        """
        Backwards-compatible accessor for detector array.
        Returns satellite detector if only satellites exist, observatory detector if only
        observatories exist, or an aggregated detector if both exist.
        """
        detectors = []
        if self.satellites and getattr(self.satellites, 'detector', None) is not None:
            detectors.append(self.satellites.detector)
        if self.observatories and getattr(self.observatories, 'detector', None) is not None:
            detectors.append(self.observatories.detector)
        if not detectors and dict.__contains__(self, '_temp_detector'):
            return self['_temp_detector']

        if not detectors:
            return None
        if len(detectors) == 1:
            return detectors[0]

        from detector import DetectorArray, appendDetector
        combined = DetectorArray(n=0)
        for d in detectors:
            appendDetector(combined, d)
        return combined

    @detector.setter
    def detector(self, value):
        """
        Backwards-compatible setter for detector.
        If satellites exist in sim_data, sets satellites.detector.
        Otherwise if observatories exist, sets observatories.detector.
        """
        if self.satellites is not None and len(self.satellites) > 0:
            self.satellites.detector = value
        elif self.observatories is not None and len(self.observatories) > 0:
            self.observatories.detector = value
        else:
            dict.__setitem__(self, '_temp_detector', value)

    def __setitem__(self, key, value):
        if key == 'detector':
            if self.satellites is not None and len(self.satellites) > 0:
                self.satellites.detector = value
            elif self.observatories is not None and len(self.observatories) > 0:
                self.observatories.detector = value
            else:
                dict.__setitem__(self, '_temp_detector', value)
            return
        cls = self._FIELD_SCHEMAS.get(key)
        if cls is not None and isinstance(value, dict) and not isinstance(value, cls):
            value = cls(**value)
        super().__setitem__(key, value)

    def __getitem__(self, key):
        if key == 'detector':
            return self.detector
        return super().__getitem__(key)

    def __contains__(self, key):
        if key == 'detector':
            return self.detector is not None
        return super().__contains__(key)

    def get_all_detectors(self):
        """
        Returns a single unified DetectorArray aggregated from all active asset components
        (satellites, observatories).
        """
        return self.detector

    def get_detector_positions(self, mask=None) -> np.ndarray:
        """
        Builds and returns an (N, 3) NumPy array of detector positions
        directly aggregated from active asset component positions.

        Args:
            mask: Optional boolean or index array to subset the output.

        Returns:
            (N, 3) array of GCRS positions for detectors.
        """
        pos_list = []
        if self.satellites and getattr(self.satellites, 'detector', None) is not None and len(self.satellites.detector) > 0:
            pos_list.append(self.satellites.position)
        if self.observatories and getattr(self.observatories, 'detector', None) is not None and len(self.observatories.detector) > 0:
            pos_list.append(self.observatories.position)

        if not pos_list:
            return np.zeros((0, 3), dtype=float)

        all_positions = np.vstack(pos_list) if len(pos_list) > 1 else pos_list[0]
        if mask is not None:
            return all_positions[mask]
        return all_positions

    def get_detector_categories(self, mask=None) -> np.ndarray:
        """
        Returns an array of category strings ('satellites', 'observatories') for all active detectors.
        """
        num_sats = len(self.satellites.detector) if (self.satellites and getattr(self.satellites, 'detector', None)) else 0
        num_obs = len(self.observatories.detector) if (self.observatories and getattr(self.observatories, 'detector', None)) else 0
        cats = np.array(['satellites'] * num_sats + ['observatories'] * num_obs)
        if mask is not None:
            return cats[mask]
        return cats


def create_empty_simulation(start_time: datetime, delta_time: float = 60.0) -> SimulationState:
    """
    Initializes a minimal, empty data structure for a space simulation.

    Args:
        start_time: The starting time and date of the simulation. This must be a
                    timezone-aware datetime object set to UTC.
        delta_time: The time step for the simulation in seconds.

    Returns:
        A SimulationState object representing the basic simulation state, not yet filled except
        for date and time and counts and pointing spheres.
    """
    if not isinstance(start_time, datetime):
        raise TypeError("start_time must be a datetime object.")
    if start_time.tzinfo is None:
        raise ValueError("start_time must be timezone-aware. Please set tzinfo.")

    return SimulationState(
        start_time=start_time,
        time=start_time,
        delta_time=delta_time
    )
