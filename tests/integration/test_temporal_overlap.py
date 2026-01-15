"""
Test suite for temporal overlap enforcement in "2 timelines" mode.

Validates the Golden Rule: "We keep ONLY timeline parts where BOTH activity and behavior exist"
"""

import numpy as np
import pytest
from iabs_synchronizer.core.alignment import (
    _calculate_temporal_overlap,
    _trim_to_temporal_overlap,
    align_data,
    _interpolate_to_timeline
)


class TestTemporalOverlapCalculation:
    """Tests for calculating temporal overlap between timelines."""

    def test_normal_overlap(self):
        """Calculate temporal overlap with normal overlap case."""
        timeline1 = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        timeline2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        overlap = _calculate_temporal_overlap(timeline1, timeline2)

        assert overlap is not None
        assert overlap[0] == 2.0
        assert overlap[1] == 5.0

    def test_no_overlap(self):
        """Calculate temporal overlap when timelines don't overlap."""
        timeline1 = np.array([0.0, 1.0, 2.0])
        timeline2 = np.array([5.0, 6.0, 7.0])

        overlap = _calculate_temporal_overlap(timeline1, timeline2)

        assert overlap is None


class TestTemporalOverlapTrimming:
    """Tests for trimming data to temporal overlap."""

    def test_trim_with_significant_data_loss(self, capsys):
        """Trimming with >5% data loss should generate warning."""
        # Data spanning 0-9s, trim to 2-5s (should lose ~60% and trigger warning)
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        timeline = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        overlap_bounds = (2.0, 5.0)

        trimmed_data, trimmed_timeline = _trim_to_temporal_overlap(
            data, timeline, overlap_bounds, feature_name="test_feature"
        )

        # Should keep points at t=2,3,4,5
        expected_length = 4
        assert len(trimmed_data) == expected_length
        assert np.allclose(trimmed_data, [3.0, 4.0, 5.0, 6.0])
        assert np.allclose(trimmed_timeline, [2.0, 3.0, 4.0, 5.0])

        # Check warning was printed
        captured = capsys.readouterr()
        assert "WARNING" in captured.out or "warning" in captured.out.lower()

    def test_trim_with_minimal_data_loss(self, capsys):
        """Trimming with <5% data loss should not generate warning."""
        # Data spanning 0-10s, trim to 0.5-9.5s (should lose <10% and not trigger warning)
        data = np.linspace(0, 10, 21)  # 21 points from 0 to 10 (step=0.5)
        timeline = np.linspace(0, 10, 21)
        overlap_bounds = (0.5, 9.5)

        trimmed_data, trimmed_timeline = _trim_to_temporal_overlap(
            data, timeline, overlap_bounds, feature_name="test_feature"
        )

        # Should keep most points
        assert len(trimmed_data) >= 17
        assert np.min(trimmed_timeline) >= 0.5
        assert np.max(trimmed_timeline) <= 9.5


class TestDirectInterpolation:
    """Tests for direct interpolation without rescaling."""

    def test_interpolation(self):
        """Test direct interpolation to target timeline."""
        # Create simple linear data
        ts = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        timeline = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        target_timeline = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

        result = _interpolate_to_timeline(ts, timeline, target_timeline)

        # Should interpolate linearly
        expected = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        assert len(result) == len(target_timeline)
        assert np.allclose(result, expected)


class TestTwoTimelinesMode:
    """Tests for '2 timelines' alignment mode."""

    def test_with_temporal_mismatch(self, capsys):
        """Test '2 timelines' mode with temporal mismatch (trimming required)."""
        # Behavior: 0-15s with 15 points
        behavior_data = np.linspace(10, 25, 15)  # Values 10-25
        behavior_timeline = np.linspace(0, 14, 15)  # 0, 1, 2, ..., 14

        # Calcium: 0-10s with 10 points
        calcium_timeline = np.linspace(0, 9, 10)  # 0, 1, 2, ..., 9

        result = align_data(
            behavior_data,
            target_length=10,
            timeline=behavior_timeline,
            target_timeline=calcium_timeline,
            mode='2 timelines'
        )

        # Result should have 10 points (calcium length)
        assert len(result) == 10

        # Verify no extrapolation beyond calcium's range happened
        assert result[-1] < 20

    def test_with_perfect_overlap(self):
        """Test '2 timelines' mode with perfect overlap (no trimming)."""
        # Both span 0-10s with different sampling
        behavior_data = np.linspace(0, 10, 40)  # 40 points
        behavior_timeline = np.linspace(0, 10, 40)

        calcium_timeline = np.linspace(0, 10, 30)  # 30 points

        result = align_data(
            behavior_data,
            target_length=30,
            timeline=behavior_timeline,
            target_timeline=calcium_timeline,
            mode='2 timelines'
        )

        # Should interpolate 40 -> 30 points without warnings
        assert len(result) == 30
        assert np.allclose(result, calcium_timeline, rtol=0.01)

    def test_with_no_overlap(self):
        """Test '2 timelines' mode with no overlap (should error)."""
        # Behavior: 0-5s
        behavior_data = np.array([1.0, 2.0, 3.0])
        behavior_timeline = np.array([0.0, 2.0, 4.0])

        # Calcium: 10-15s (no overlap)
        calcium_timeline = np.array([10.0, 12.0, 14.0])

        with pytest.raises(ValueError) as exc_info:
            align_data(
                behavior_data,
                target_length=3,
                timeline=behavior_timeline,
                target_timeline=calcium_timeline,
                mode='2 timelines'
            )

        assert "No temporal overlap" in str(exc_info.value)

    def test_behavior_extends_past_calcium(self, capsys):
        """Test '2 timelines' mode with behavior extending past calcium."""
        # Behavior: 0-20s @ 40fps (800 points) - extends beyond calcium
        behavior_data = np.sin(np.linspace(0, 4*np.pi, 800))  # Some sinusoidal signal
        behavior_timeline = np.linspace(0, 20, 800)

        # Calcium: 0-10s @ 30fps (300 points)
        calcium_timeline = np.linspace(0, 10, 300)

        result = align_data(
            behavior_data,
            target_length=300,
            timeline=behavior_timeline,
            target_timeline=calcium_timeline,
            mode='2 timelines'
        )

        # Result should match calcium length
        assert len(result) == 300

        # Verify result is reasonable (sin values should be in [-1, 1])
        assert np.max(result) <= 1.5
        assert np.min(result) >= -1.5

    def test_calcium_extends_past_behavior(self, capsys):
        """Test '2 timelines' mode with calcium extending past behavior."""
        # Behavior: 0-8s (shorter than calcium)
        behavior_data = np.linspace(100, 200, 160)  # 160 points
        behavior_timeline = np.linspace(0, 8, 160)

        # Calcium: 0-10s (extends beyond behavior)
        calcium_timeline = np.linspace(0, 10, 300)

        result = align_data(
            behavior_data,
            target_length=300,
            timeline=behavior_timeline,
            target_timeline=calcium_timeline,
            mode='2 timelines'
        )

        # Result should be trimmed to behavior's range (0-8s)
        assert len(result) < 300
        assert len(result) > 200
