# Cadence Controller Implementation Plan

This document outlines the steps to implement a cadence controller for VibeVolts, allowing detectors with different integration times to be processed at their respective intervals.

## 0. Version Control Setup
- Create a new git branch: `git checkout -b feature/cadence-controller`.
- This ensures all changes are isolated until verified.

## 1. Modify `scandetectors.py`


Do remove any of the docuumentation in the file, but modify or add if the change in mask makes a difference. 

In particular each of the 6 steps  defined in the first block should still exist, although relevant references to masks maybe added.

Update `scandetectors` to use NumPy vectorization for both detectors and targets.

- **Signature Update:** `def scandetectors(sim_data: dict, print_output: int = 0, mask: np.ndarray = None):`
- **Masking:** If `mask` is provided, subset all detector-related arrays (positions, pointings, FOVs, etc.) at the start of the function. If `mask` is None, use all detectors.
- **Vectorized Geometry:**
    - Use broadcasting to calculate vectors from all active satellites to all targets: `toTargets = targets[np.newaxis, :, :] - active_sat_positions[:, np.newaxis, :]` (Shape: `(num_active_sats, num_targets, 3)`).
    - Calculate angles between detector pointing vectors and `toTargets` using vectorized `np.einsum` and `np.arccos`.
- **Visibility Detection:** Create a 2D boolean mask `visible_mask = angles < active_fovs[:, np.newaxis]` (Shape: `(num_active_sats, num_targets)`).
- **Target Radiometry:**
    - Use `active_indices, target_indices = np.where(visible_mask)` to identify specific satellite-target pairs for processing.
    - Perform all subsequent calculations (Lambertian brightness, flux, signal, noise, SNR) using these indexed pairs in a fully vectorized manner.
- **Output:** Ensure printing logic (if enabled) iterates through the results efficiently.

## 2. Update `propagation.py` Usage
The existing `propagate_satellites` function already updates all satellites in the specified categories. No functional changes are needed for this file, but the `nextIntegration` logic must ensure it calls this function for the entire constellation to maintain a consistent state.

## 3. Create `cadenceController.py`
Implement the logic for managing detector groups.

### `initCadence(sim_data)`
1.  Access `sim_data['detector'].integrationTime`.
2.  Find unique integration times using `np.unique`.
3.  Initialize `sim_data['cadenceStructure']` as a dictionary.
4.  `cadenceList = []`
5.  For each unique interval:
    - Create a `cadenceGroup` dictionary:
        - `scanInterval`: The unique interval value.
        - `scanMask`: `sim_data['detector'].integrationTime == interval`.
        - `scanNext`: Initialize to `sim_data['time']` so all groups perform an initial scan.
    - Append to `cadenceList`.
6.  Set `sim_data['cadenceStructure']['cadenceList'] = cadenceList`.
7.  Find the group with the minimum `scanNext` and set `sim_data['cadenceStructure']['nextTime']` and `sim_data['cadenceStructure']['nextGroup']`.

### `nextIntegration(sim_data)`
1.  Get `cadenceStructure` from `sim_data`.
2.  Set `sim_data['time'] = cadenceStructure['nextTime']`.
3.  **Propagate ALL satellites:** Call `propagate_satellites(sim_data, sim_data['time'])`.
4.  Get the group at index `cadenceStructure['nextGroup']`.
5.  Get `mask = group['scanMask']`.
6.  **Scan Group (Vectorized):** Call `scandetectors(sim_data, mask=mask)`.
7.  Update group's `scanNext`: `group['scanNext'] = sim_data['time'] + timedelta(seconds=group['scanInterval'])`.
8.  Find the group with the earliest `scanNext` in `cadenceList`.
9.  Update `cadenceStructure['nextTime']` and `cadenceStructure['nextGroup']`.

## 4. Verification
- Create a test script with detectors having different integration times.
- Run `nextIntegration` multiple times and verify that:
    - `sim_data['time']` advances correctly.
    - All satellites are propagated.
    - `scandetectors` processes only the correct masked subset of detectors.
    - Vectorized calculations produce correct SNR results for visible targets.
