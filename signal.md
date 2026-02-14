# Tracing the 'signal' variable in `scandetectors.py`

The `signal` variable in `scandetectors.py` represents the detected flux of light from a target over a given integration time and detector area. It is calculated as follows:

```python
signal = detectorFlux * integrationTime[i] * detectorArea[i]
```
*(Source: `scandetectors.py`)*

Let's trace each of the contributing variables, starting from the most primitive elements.

## Primitive Elements and Constants

### `ARCSEC` Constant

`ARCSEC` is a constant imported from `constants.py`.

```python
ARCSEC = 2.0 *pi/(360*3600)
```
*(Source: `constants.py`)*

This constant converts arcseconds to radians. Specifically, it's `2 * pi / (360 degrees * 3600 arcseconds/degree)`.

### `amag` Function Definition

```python
def amag(x: float) -> float:
    """
    Calculates the linear ratio from a magnitude value.
    This is the inverse of the mag() function.
    Uses the formula: ratio = 10**(-0.4 * magnitude)
    """
    return 10**(-0.4 * x)
```
*(Source: `radiometry_calcs.py`)*

The `amag` function converts an apparent magnitude value (`x`) to a linear ratio using the formula `10**(-0.4 * magnitude)`.

### `FILTER_DATA` Structure

`FILTER_DATA` is a dictionary imported from `radiometry_data.py`. It contains data for various astronomical filter bands (e.g., 'U', 'B', 'V', 'J', 'H', 'K', 'g', 'r', 'i', 'z', 'F560W', etc.). Each band is a dictionary with keys such as:
*   `'sun'`: Apparent magnitude of the sun in the given filter band.
*   `'sky'`: Sky brightness at a dark site in magnitudes per square arcsecond.
*   `'space'`: Sky brightness in space.
*   `'central_wavelength'`: Central wavelength in nanometers (nm).
*   `'bandwidth'`: Bandwidth in nanometers (nm).
*   `'zero_point'`: Photon flux for a 0-magnitude object in photons per second per square meter.

*(Source: `radiometry_data.py`)*

### Simulation Data Inputs

These are the direct extractions from the `sim_data` dictionary or derived from it:

*   **`sim_data['detector'].filt[0]`**: This value represents the astronomical filter band being used by the detector (e.g., "V", "J", "F560W"). It is a string that acts as a key to look up data in the `FILTER_DATA` dictionary. This `filt` attribute of the `detector` object within `sim_data` is set when a detector is initialized, typically by the `makeDetector` function in `detector.py`.
*   **`albedo` (`sim_data['fixedpoints']['albedo']`)**: This is the albedo (reflectivity) of the fixed targets.
*   **`radius` (`sim_data['fixedpoints']['size']/2`)**: The radius of the fixed targets, derived from their diameter.
*   **`sunVect` (`sim_data['celestial']['position'][0]`)**: Represents the position vector of the Sun relative to the simulation origin.
*   **`satpositions` (`sim_data['satellites']['position']`)**: Positions of all satellites.
*   **`targets` (`sim_data['fixedpoints']['position']`)**: Positions of all fixed targets.
*   **`apertureArea` (`sim_data['detector'].apertureArea`)**: The physical area of the detector's aperture, typically set in `makeDetector` based on the `aper` (aperture diameter) argument.
*   **`integrationTime` (`sim_data['detector'].integrationTime`)**: The duration over which the detector collects light. Initially an array of zeros, but typically populated by `requiredIntegrationTime` in `detector.py`.

## Intermediate Calculations

### `radiometry_calcs.fluxes` Function

This function uses the `FILTER_DATA`, `amag` function, and `ARCSEC` constant along with the `band` (from `sim_data['detector'].filt[0]`) to calculate the `sun` variable.

```python
def fluxes(band):
    """
    uses the FILTER_DATA table from radiometry_data.py for data
    Looks up in formation based on the argument band, which is
    usually something like an astronomical band... U, B, V, etc.
    It returns three numbers:
    sun which is the solar flux at earth in photons/s/m^2
    sky which is the sky brightness at earth in p/s/asec^2/m^2
    space, sky brightness in space in p/s/asec^2/m^2
    """
    x = FILTER_DATA[band]
    zp = x['zero_point']
    sun = amag(x['sun']) * zp
    space = amag(x['space']) * zp / (ARCSEC**2)
    sky = amag(x['sky']) * zp / (ARCSEC**2)
    return(sun, space, sky)
```
*(Source: `radiometry_calcs.py`)*

The `sun` value returned by this function represents the solar flux at Earth in photons/s/m^2 for a given `band`. It is calculated as:

```python
sun = amag(x['sun']) * zp
```

### `sun` Variable

The `sun` variable (used in `lambertian.lambertiansphere`) is one of the outputs from the `radiometry_calcs.fluxes` function:

```python
sun, space, sky = radiometry_calcs.fluxes(sim_data['detector'].filt[0])
```
*(Source: `scandetectors.py`)*

### `toTargets` and `mask`

These variables define the geometry and visibility of targets relative to the satellite.

*   **`toTargets`**: The vector from the satellite to the targets:
    ```python
    toTargets = targets - satposition
    ```
    Where `targets` are `sim_data['fixedpoints']['position']` and `satposition` is `satpositions[i, :]`.
*   **`mask`**: A boolean array that filters targets based on the detector's Field of View (FOV).
    ```python
    angles = np.arccos(np.clip(dot_products /
                               (norms_V * norms_W), -1.0, 1.0))
    fov = fovs[i]
    mask = angles < fov
    ```
    This involves geometric calculations of angles between the detector's pointing vector and the vectors to the targets.

### `lambertian.lambertiansphere` Function

This function calculates the apparent brightness of diffusely reflecting spheres. Its output, `apparent_brightness`, is determined by several inputs, including the `sun` variable, `albedo`, `radius`, `sunVect`, and `toTargets[mask]`.

```python
def lambertiansphere(
    vec_from_sphere_to_light: np.ndarray,
    vec_from_sphere_to_observer: np.ndarray,
    albedo: np.ndarray,
    radius: np.ndarray,
    base_brightness: np.ndarray
) -> np.ndarray:
    """
    Calculates the illuminance of multiple lambertian spheres.
    If base brightness is given in photons/m^2, the result will be in the
    same units at the observer defined by the specified geometry
    ...
    """
```
*(Source: `lambertian.py`)*

The `lambertiansphere` function uses:
*   `vec_from_sphere_to_light`: Corresponds to `-sunVect` (vector from target to Sun).
*   `vec_from_sphere_to_observer`: Corresponds to `-toTargets[mask]` (vector from target to observer).
*   `albedo`: `albedo[mask]` (reflectivity of visible targets).
*   `radius`: `radius[mask]` (radius of visible targets).
*   `base_brightness`: `sun` (solar flux).

The core calculation within `lambertiansphere` is:
```python
    apparent_brightness = (
        (base_brightness * effective_cross_section) /
        (np.pi * norm_observer ** 2)
    )
```
Where `effective_cross_section` is derived from `albedo`, `radius`, and a `phase_function_value`, and `norm_observer` is the magnitude of the `vec_from_sphere_to_observer`.

## Components of the `signal` Calculation

### 1. `detectorFlux`

`detectorFlux` is the incident flux on the detector from the target. It is the output of the `lambertiansphere` function:

```python
detectorFlux = lambertian.lambertiansphere(
     -sunVect,
     -toTargets[mask],
     albedo[mask],
     radius[mask],
     sun)
```
*(Source: `scandetectors.py`)*

### 2. `integrationTime[i]`

`integrationTime` is extracted from the simulation data:

```python
integrationTime = sim_data['detector'].integrationTime
```
*(Source: `scandetectors.py`)*

The `[i]` selects the integration time specific to the current satellite. This attribute is typically set by the `requiredIntegrationTime` function in `detector.py`.

### 3. `detectorArea[i]`

`detectorArea` is extracted from the simulation data:

```python
detectorArea = sim_data['detector'].apertureArea
```
*(Source: `scandetectors.py`)*

This represents the physical area of the detector's aperture. The `[i]` selects the aperture area specific to the current satellite (if `detectorArea` is an array).

## Conclusion: The `signal` Calculation

Finally, the `signal` variable is calculated by multiplying these three components:

```python
signal = detectorFlux * integrationTime[i] * detectorArea[i]
```
*(Source: `scandetectors.py`)*

## Potential Discrepancy in `requiredIntegrationTime` Formula

A potential issue has been identified in the `requiredIntegrationTime` function within `detector.py`, which is responsible for populating `sim_data['detector'].integrationTime`. The docstring for this function defines the Signal-to-Noise Ratio (SNR) as `SNR = (Signal) / sqrt(Signal + Background)`, where `Signal` and `Background` are understood to be total photon counts over the integration time `t`.

Based on this definition, a standard derivation for the required integration time `t` would typically yield:

`t = SNR^2 * (alpha + beta * omega) / (alpha^2 * A * eta * f)`

However, the formula stated in the `requiredIntegrationTime` docstring and implemented in the codebase is:

`t = SNR^2 * beta * omega /( alpha^2 * A * eta * f)`

Comparing these two formulas reveals a significant discrepancy in the numerator:

1.  **Missing Signal Term in Numerator:** The implemented formula (and its corresponding docstring) uses `beta * omega`, whereas a direct derivation from the stated SNR definition would include `(alpha + beta * omega)`. This difference suggests that the implemented formula might be neglecting the signal contribution to the noise term, or it could be an approximation where background noise heavily dominates, or it may be based on a different underlying definition of SNR than stated in the docstring.

It is important to note that the `f^2` versus `f` discrepancy previously identified has been resolved; the current implementation uses `f` as expected.

This inconsistency between the stated SNR definition in the docstring and the implemented formula for calculating `t` could lead to an inaccurate determination of the required integration time. If the implemented formula is indeed based on a different SNR definition or a specific approximation, the docstring should be updated to clarify these assumptions. Conversely, if the intent is to strictly adhere to the stated SNR definition, the formula in the code may need to be revised. This discrepancy could directly impact the radiometric performance estimates of the simulation.
