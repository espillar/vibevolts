import numpy as np
from datetime import datetime
from typing import Any
from dataclasses import dataclass


def safe_normalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Safely normalizes an array of vectors along the specified axis,
    preventing division by zero.
    """
    norms = np.linalg.norm(v, axis=axis, keepdims=True)
    return np.where(norms == 0, 0.0, v / norms)


class DotDict(dict):
    """
    Dictionary subclass that supports dot-notation attribute access alongside
    standard dictionary lookup and mutation.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    def __setattr__(self, name, value):
        self[name] = value

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, AttributeError):
            return default


SchemaDict = DotDict


class ComponentState(DotDict):
    """
    Base class for component sub-states stored as parallel NumPy arrays and lists
    indexed along axis 0 by asset count N.
    """

    def __len__(self) -> int:
        for val in self.values():
            if isinstance(val, (np.ndarray, list)):
                return len(val)
        return 0

    def subset(self, mask: Any) -> "ComponentState":
        """
        Slices all array, list, and nested ComponentState fields by `mask`
        along axis 0 and returns a new instance of the same class.
        """
        indices = np.arange(len(self))[mask]
        new_kwargs = {}
        for key, val in self.items():
            if isinstance(val, np.ndarray):
                new_kwargs[key] = val[mask]
            elif isinstance(val, list):
                new_kwargs[key] = [val[i] for i in indices]
            elif isinstance(val, ComponentState):
                new_kwargs[key] = val.subset(mask)
            else:
                new_kwargs[key] = val
        return self.__class__(**new_kwargs)

    def append(self, other: "ComponentState") -> None:
        """
        Appends matching array, list, and nested ComponentState fields from
        `other` in-place along axis 0.
        """
        for key, val in other.items():
            if key not in self or self[key] is None:
                self[key] = val
            elif isinstance(self[key], np.ndarray) and isinstance(val, np.ndarray):
                self[key] = np.concatenate([self[key], val], axis=0)
            elif isinstance(self[key], list) and isinstance(val, list):
                self[key].extend(val)
            elif isinstance(self[key], ComponentState) and isinstance(val, ComponentState):
                self[key].append(val)


class CountsState(DotDict):
    """
    Provides live asset counts backed by the parent SimulationState.
    """
    def __init__(self, parent_sim=None, **kwargs):
        object.__setattr__(self, '_parent_sim', parent_sim)
        super().__init__(
            celestial=0,
            satellites=0,
            observatories=0,
            fixedpoints=0,
            **kwargs,
        )

    def _live_count(self, item):
        parent = self._parent_sim
        if parent is not None and item in parent:
            comp = parent[item]
            if comp is not None and hasattr(comp, '__len__'):
                return len(comp)
        return None

    def __getitem__(self, item):
        count = self._live_count(item)
        if count is not None:
            return count
        return super().__getitem__(item)

    def __contains__(self, item):
        return super().__contains__(item) or self._live_count(item) is not None

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    def values(self):
        return [self[k] for k in self.keys()]


class CelestialState(ComponentState):
    def __init__(self, **kwargs):
        defaults = {
            'position': np.zeros((2, 3)),
            'velocity': np.zeros((2, 3)),
        }
        super().__init__(**{**defaults, **kwargs})


class SatellitesState(ComponentState):
    def __init__(self, **kwargs):
        defaults = {
            'position': np.zeros((0, 3)),
            'velocity': np.zeros((0, 3)),
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
                      is due. Initialised to sim_data.time so that every
                      group fires on the first call to nextIntegration.
    """
    scanInterval: float
    scanMask: np.ndarray
    scanNext: datetime


class CadenceState(DotDict):
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


class SimulationState(DotDict):
    """
    Main simulation data structure supporting both dictionary and attribute access.
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

    def _active_detector_components(self):
        """Returns [(category_name, component), ...] for components with non-empty detectors."""
        active = []
        for name in ('satellites', 'observatories'):
            comp = dict.__getitem__(self, name) if dict.__contains__(self, name) else None
            if comp is not None and getattr(comp, 'detector', None) is not None and len(comp.detector) > 0:
                active.append((name, comp))
        return active

    @property
    def detector(self):
        """
        Backwards-compatible accessor for detector array.
        Returns satellite detector if only satellites exist, observatory detector if only
        observatories exist, or an aggregated detector if both exist.
        """
        detectors = [comp.detector for _, comp in self._active_detector_components()]
        if not detectors and dict.__contains__(self, '_temp_detector'):
            return dict.__getitem__(self, '_temp_detector')
        if not detectors:
            return None
        if len(detectors) == 1:
            return detectors[0]

        from detector import DetectorArray
        combined = DetectorArray(n=0)
        for d in detectors:
            combined.append(d)
        return combined

    @detector.setter
    def detector(self, value):
        self._set_detector(value)

    def _set_detector(self, value):
        if self.satellites is not None and len(self.satellites) > 0:
            self.satellites.detector = value
        elif self.observatories is not None and len(self.observatories) > 0:
            self.observatories.detector = value
        else:
            dict.__setitem__(self, '_temp_detector', value)

    def __setitem__(self, key, value):
        if key == 'detector':
            self._set_detector(value)
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
        pos_list = [comp.position for _, comp in self._active_detector_components()]
        if not pos_list:
            return np.zeros((0, 3), dtype=float)
        all_positions = np.vstack(pos_list) if len(pos_list) > 1 else pos_list[0]
        return all_positions[mask] if mask is not None else all_positions

    def get_detector_categories(self, mask=None) -> np.ndarray:
        """
        Returns an array of category strings ('satellites', 'observatories')
        for all active detectors.
        """
        cats_list = []
        for name, comp in self._active_detector_components():
            cats_list.extend([name] * len(comp.detector))
        cats = np.array(cats_list)
        return cats[mask] if mask is not None else cats


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
        delta_time=delta_time,
    )
