"""
Test what happens in "2 timelines" mode when both timelines have gaps.

These tests explore edge cases where both calcium and behavior timelines
contain recording gaps (non-continuous recording).
"""

import numpy as np
import pytest
from iabs_synchronizer.core.alignment import align_data


class TestBothTimelinesWithGaps:
    """Tests for '2 timelines' mode when both timelines have gaps."""

    def test_both_timelines_with_gaps(self):
        """Test when both behavior and calcium have gaps."""
        # Behavior timeline: recorded at 0-3s, then GAP, then 6-9s
        behavior_timeline = np.array([0.0, 1.0, 2.0, 3.0,  # First segment
                                     6.0, 7.0, 8.0, 9.0])  # Second segment after gap

        # Behavior data: some signal values
        behavior_data = np.array([10, 20, 30, 40,  # First segment values
                                 60, 70, 80, 90])  # Second segment values

        # Calcium timeline: recorded at 1-4s, then GAP, then 7-10s
        calcium_timeline = np.array([1.0, 2.0, 3.0, 4.0,  # First segment
                                    7.0, 8.0, 9.0, 10.0])  # Second segment after gap

        # Run alignment
        result = align_data(
            behavior_data,
            target_length=len(calcium_timeline),
            timeline=behavior_timeline,
            target_timeline=calcium_timeline,
            mode='2 timelines'
        )

        # Temporal overlap is [1.0, 9.0], which trims calcium points at t=10.0
        # Result length will be trimmed calcium timeline within overlap
        assert len(result) < len(calcium_timeline)
        assert len(result) >= 6

        # Note: Current implementation interpolates through gaps
        # This is expected behavior for the current design


class TestBehaviorInCalciumGap:
    """Tests for when behavior data exists only in calcium's gap."""

    def test_behavior_only_in_calcium_gap(self):
        """Test when all behavior data is in calcium's gap."""
        # Behavior: 4-6s (only in calcium's gap)
        behavior_timeline = np.array([4.0, 4.5, 5.0, 5.5, 6.0])
        behavior_data = np.array([100, 110, 120, 130, 140])

        # Calcium: 0-3s, GAP 3-7s, 7-10s (gap where behavior exists)
        calcium_timeline = np.array([0.0, 1.0, 2.0, 3.0,  # Before gap
                                    7.0, 8.0, 9.0, 10.0])  # After gap

        # Current implementation: temporal overlap is calculated as [4.0, 6.0]
        # After trimming calcium timeline to this range, no calcium points remain
        # This results in an empty array being returned
        result = align_data(
            behavior_data,
            target_length=len(calcium_timeline),
            timeline=behavior_timeline,
            target_timeline=calcium_timeline,
            mode='2 timelines'
        )

        # Document current behavior: returns empty array when no calcium points in overlap
        assert len(result) == 0
        assert isinstance(result, np.ndarray)


class TestPartialOverlapWithGaps:
    """Tests for realistic scenarios with partial overlap and gaps."""

    def test_realistic_scenario_with_partial_overlap(self):
        """Test realistic scenario: behavior has gap, calcium is continuous."""
        # Behavior: 0-5s, GAP, 10-15s
        behavior_timeline = np.concatenate([
            np.linspace(0, 5, 6),
            np.linspace(10, 15, 6)
        ])
        behavior_data = np.concatenate([
            np.linspace(100, 150, 6),  # Segment 1
            np.linspace(200, 250, 6)   # Segment 2
        ])

        # Calcium: 0-12s continuous
        calcium_timeline = np.linspace(0, 12, 13)

        result = align_data(
            behavior_data,
            target_length=len(calcium_timeline),
            timeline=behavior_timeline,
            target_timeline=calcium_timeline,
            mode='2 timelines'
        )

        assert len(result) == len(calcium_timeline)

        # Note: Calcium points in 5-10s range will be interpolated from behavior edges
        # This is current expected behavior (interpolates through gaps)
