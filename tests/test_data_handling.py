import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from dataHandling import DataHandler
import plotly.graph_objects as go

def test_data_handler_empty():
    """Test gap time calculation on an empty DataHandler."""
    handler = DataHandler()
    assert handler.calculate_gap_times(target_id=1) == pytest.approx([])
    assert handler.calculate_gap_times() == {}
    assert handler.calculate_gap_times(pooled=True) == pytest.approx([])

    # Test plotting on empty handler
    fig = handler.plot_gap_times_histogram(show_plot=False)
    assert isinstance(fig, go.Figure)
    assert "No Gap Data Available" in fig.layout.title.text

def test_data_handler_single_observations():
    """Test targets with only a single observation (no gaps)."""
    handler = DataHandler()
    t1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    results = {
        'time': t1,
        'sat_indices': np.array([0, 1]),
        'target_indices': np.array([10, 20]),
        'signal': np.array([100.0, 200.0]),
        'noise': np.array([10.0, 15.0]),
        'snr': np.array([10.0, 13.3])
    }
    handler.add_results(results)

    # With only 1 observation per target, gap times should be empty
    assert len(handler.calculate_gap_times(target_id=10)) == 0
    assert len(handler.calculate_gap_times(target_id=20)) == 0
    gaps = handler.calculate_gap_times()
    assert 10 in gaps and len(gaps[10]) == 0
    assert 20 in gaps and len(gaps[20]) == 0
    assert len(handler.calculate_gap_times(pooled=True)) == 0

    fig = handler.plot_gap_times_histogram(show_plot=False)
    assert isinstance(fig, go.Figure)
    assert "No Gap Data Available" in fig.layout.title.text

def test_data_handler_multiple_observations():
    """Test targets with multiple observations (valid gaps)."""
    handler = DataHandler()
    t1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    t2 = t1 + timedelta(seconds=15)
    t3 = t1 + timedelta(seconds=45)

    # Step 1
    handler.add_results({
        'time': t1,
        'sat_indices': np.array([0, 1]),
        'target_indices': np.array([10, 20]),
        'signal': np.array([100.0, 200.0]),
        'noise': np.array([10.0, 15.0]),
        'snr': np.array([10.0, 13.3])
    })

    # Step 2
    handler.add_results({
        'time': t2,
        'sat_indices': np.array([0, 1]),
        'target_indices': np.array([10, 20]),
        'signal': np.array([110.0, 210.0]),
        'noise': np.array([10.0, 15.0]),
        'snr': np.array([11.0, 14.0])
    })

    # Step 3 (Only target 10 is observed)
    handler.add_results({
        'time': t3,
        'sat_indices': np.array([0]),
        'target_indices': np.array([10]),
        'signal': np.array([120.0]),
        'noise': np.array([10.0]),
        'snr': np.array([12.0])
    })

    # Target 10 observed at t1, t2, t3: gaps = [15s, 30s]
    gaps_10 = handler.calculate_gap_times(target_id=10)
    np.testing.assert_allclose(gaps_10, [15.0, 30.0])

    # Target 20 observed at t1, t2: gaps = [15s]
    gaps_20 = handler.calculate_gap_times(target_id=20)
    np.testing.assert_allclose(gaps_20, [15.0])

    # Dict lookup
    gaps_dict = handler.calculate_gap_times()
    assert list(gaps_dict.keys()) == [10, 20]
    np.testing.assert_allclose(gaps_dict[10], [15.0, 30.0])
    np.testing.assert_allclose(gaps_dict[20], [15.0])

    # Pooled lookup: [15.0, 30.0, 15.0]
    gaps_pooled = handler.calculate_gap_times(pooled=True)
    # The order depends on groupby sorting; target 10 comes before target 20
    np.testing.assert_allclose(gaps_pooled, [15.0, 30.0, 15.0])

    # Test plotting
    fig = handler.plot_gap_times_histogram(show_plot=False)
    assert isinstance(fig, go.Figure)
    assert "Pooled Interobservation Gap Times" in fig.layout.title.text

    fig_single = handler.plot_gap_times_histogram(target_id=10, show_plot=False)
    assert isinstance(fig_single, go.Figure)
    assert "Target 10" in fig_single.layout.title.text
