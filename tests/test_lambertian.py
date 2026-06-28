import numpy as np
import pytest
from io import StringIO
import sys

from lambertian import lambertiansphere, simple_lambertian, includedAngle

def test_lambertiansphere_debug_level_2(capsys):
    """
    Test lambertiansphere function with debug level 2 to ensure correct output.
    """
    angle_light_observer = np.array([0.5, np.pi / 2, 2.5])
    albedo = np.array([0.1, 0.5, 0.9])
    radius = np.array([100.0, 500.0, 1000.0])
    base_brightness = np.array([1.0, 10.0, 100.0])

    # Call the function with debug=2
    lambertiansphere(angle_light_observer, albedo, radius, base_brightness, debug=2)

    # Capture the output
    captured = capsys.readouterr()
    output = captured.out

    # Assert that key elements of the debug output are present
    assert "--- Detailed Debug Info: lambertiansphere ---" in output
    assert "Input Angle" in output
    assert "Clipped Alpha" in output
    assert "Phase Func Val" in output
    assert "Cross Sect Area" in output
    assert "Effective CS" in output
    assert "Emitted Brightness" in output

    # Check for specific values from the first sphere's data
    # (these values are illustrative and would need to be precisely calculated
    # if a more rigorous test of the numerical output were desired)
    assert f"{angle_light_observer[0]:<15.4e}" in output
    assert f"{albedo[0]:<10.4f}" in output
    assert f"{radius[0]:<12.4e}" in output
    assert f"{base_brightness[0]:<18.4e}" in output

    # Test debug level 1 still works
    lambertiansphere(angle_light_observer, albedo, radius, base_brightness, debug=1)
    captured_debug1 = capsys.readouterr()
    output_debug1 = captured_debug1.out
    assert "--- Debug Info: lambertiansphere ---" in output_debug1
    assert "Effective Cross Section:" in output_debug1
    assert "Phase Angle (rad)" in output_debug1
    assert "Emitted Brightness" in output_debug1
    assert "Input Angle" not in output_debug1 # Ensure debug 2 specific output is not in debug 1


def test_lambertian_consistency():
    """
    Verify that simple_lambertian and lambertiansphere are mathematically consistent
    when the division factor of pi * distance^2 is applied.
    """
    diameter = 3.0
    radius = diameter / 2.0
    distance = 1000e3
    albedo = 0.3
    angle = np.pi / 2.0
    base_brightness = 1361.0

    val_simple = simple_lambertian(
        diameter=diameter,
        distance=distance,
        albedo=albedo,
        angle=angle,
        base_brightness=base_brightness
    )

    emitted = lambertiansphere(
        angle_light_observer=np.array([angle]),
        albedo=np.array([albedo]),
        radius=np.array([radius]),
        base_brightness=np.array([base_brightness])
    )[0]
    val_sphere = emitted / (np.pi * distance**2)

    assert np.allclose(val_simple, val_sphere)


