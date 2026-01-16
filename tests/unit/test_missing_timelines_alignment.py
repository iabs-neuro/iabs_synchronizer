"""
Test alignment behavior when time series don't have associated timelines.

Unit tests for alignment logic with missing timelines:
1. Behavior has no timeline, Calcium has timeline
2. Calcium has no timeline (unusual but possible)
3. Both have no timelines
"""

import numpy as np
import pytest
from iabs_synchronizer.core.alignment import align_data, align_all_data


class TestMissingBehaviorTimeline:
    """Tests for when behavior timeline is missing."""

    def test_no_behavior_timeline_matching_lengths(self):
        """When behavior has no timeline but lengths match, should use 'simple' mode."""
        behavior_data = np.linspace(0, 10, 100)
        behavior_timeline = None  # No timeline

        calcium_timeline = np.linspace(0, 10, 100)
        target_length = 100

        result = align_data(behavior_data, target_length, behavior_timeline,
                           calcium_timeline, mode='simple')

        assert len(result) == 100
        assert np.array_equal(result, behavior_data)

    def test_no_behavior_timeline_different_lengths_raises_error(self):
        """When behavior has no timeline and lengths differ (but calcium has timeline), should raise error."""
        behavior_data = np.linspace(0, 10, 80)  # Different length
        behavior_timeline = None

        calcium_timeline = np.linspace(0, 10, 100)
        target_length = 100

        # Only one timeline present - suspicious configuration
        filtered_data = {
            'Calcium': {
                'Calcium': np.random.rand(10, 100),
                'timeline': calcium_timeline,  # HAS timeline
                'fps': 10
            },
            'Behavior_auto': [
                {
                    'Speed': behavior_data,
                    'timeline': None,  # MISSING timeline
                    'fps': None
                }
            ]
        }

        # Should raise error
        with pytest.raises(ValueError) as exc_info:
            align_all_data(filtered_data, force_pathway=None)

        error_msg = str(exc_info.value)
        assert "Only target timeline present" in error_msg or "suspicious" in error_msg.lower()

    def test_no_behavior_timeline_forced_mode_works(self):
        """Forced mode should work when automatic selection fails."""
        behavior_data = np.linspace(0, 10, 80)
        behavior_timeline = None

        calcium_timeline = np.linspace(0, 10, 100)

        filtered_data = {
            'Calcium': {
                'Calcium': np.random.rand(10, 100),
                'timeline': calcium_timeline,
                'fps': 10
            },
            'Behavior_auto': [
                {
                    'Speed': behavior_data,
                    'timeline': None,
                    'fps': None
                }
            ]
        }

        # Force cast_to_ca mode should work
        aligned_data, align_log, mode_stats = align_all_data(filtered_data, force_pathway='cast_to_ca')

        feature_keys = {k.lower() for k in aligned_data.keys()}
        assert 'speed' in feature_keys
        assert 'cast_to_ca' in mode_stats


class TestMissingCalciumTimeline:
    """Tests for when calcium timeline is missing (unusual case)."""

    def test_no_calcium_timeline_matching_lengths(self):
        """When calcium has no timeline but lengths match, should use 'simple' mode."""
        behavior_data = np.linspace(0, 10, 100)
        behavior_timeline = np.linspace(0, 10, 100)

        calcium_timeline = None  # No calcium timeline
        target_length = 100

        # Automatic mode should fall back to 'simple'
        filtered_data = {
            'Calcium': {
                'Calcium': np.random.rand(10, 100),
                'timeline': None,  # No timeline
                'fps': None
            },
            'Behavior_auto': [
                {
                    'Speed': behavior_data,
                    'timeline': behavior_timeline,
                    'fps': 10
                }
            ]
        }

        aligned_data, align_log, mode_stats = align_all_data(filtered_data, force_pathway=None)

        feature_keys = {k.lower() for k in aligned_data.keys()}
        assert 'speed' in feature_keys


class TestBothTimelinessMissing:
    """Tests for when both timelines are missing."""

    def test_both_missing_matching_lengths(self):
        """When both timelines missing but lengths match, should use 'simple' mode."""
        behavior_data = np.linspace(0, 10, 100)
        behavior_timeline = None

        calcium_timeline = None
        target_length = 100

        result = align_data(behavior_data, target_length, behavior_timeline,
                           calcium_timeline, mode='simple')

        assert len(result) == 100
        assert np.array_equal(result, behavior_data)

    def test_both_missing_different_lengths_uses_cast_to_ca(self):
        """When both timelines missing and lengths differ, should use cast_to_ca fallback."""
        behavior_data = np.linspace(0, 10, 80)
        behavior_timeline = None

        filtered_data = {
            'Calcium': {
                'Calcium': np.random.rand(10, 100),
                'timeline': None,
                'fps': None
            },
            'Behavior_auto': [
                {
                    'Speed': behavior_data,
                    'timeline': None,
                    'fps': None
                }
            ]
        }

        # Should succeed with cast_to_ca fallback (synthetic timelines created)
        aligned_data, log, mode_stats = align_all_data(filtered_data, force_pathway=None)
        assert 'Speed' in aligned_data
        assert len(aligned_data['Speed']) == 100
        assert 'cast_to_ca' in mode_stats

    def test_both_missing_forced_crop_mode(self):
        """Forced crop mode should work when both timelines missing."""
        behavior_data = np.linspace(0, 10, 120)  # Longer than calcium
        behavior_timeline = None

        filtered_data = {
            'Calcium': {
                'Calcium': np.random.rand(10, 100),
                'timeline': None,
                'fps': None
            },
            'Behavior_auto': [
                {
                    'Speed': behavior_data,
                    'timeline': None,
                    'fps': None
                }
            ]
        }

        # Force crop mode should work
        aligned_data, align_log, mode_stats = align_all_data(filtered_data, force_pathway='crop')

        feature_keys = {k.lower() for k in aligned_data.keys()}
        assert 'speed' in feature_keys
        # Speed should be cropped to 100 points
        speed_key = [k for k in aligned_data.keys() if k.lower() == 'speed'][0]
        assert len(aligned_data[speed_key]) == 100
        assert 'crop' in mode_stats
