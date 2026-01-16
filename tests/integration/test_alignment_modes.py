"""
Test suite for alignment mode fixes.

Tests:
1. 'simple' mode validates length equality
2. 'crop' mode handles equality case correctly
3. Error thrown when no alignment method is suitable
"""

import numpy as np
import pytest
from iabs_synchronizer.core.alignment import align_data, align_all_data


class TestSimpleMode:
    """Tests for 'simple' alignment mode."""

    def test_simple_mode_accepts_matching_lengths(self):
        """'simple' mode should accept matching lengths."""
        behavior_data = np.arange(100)
        result = align_data(behavior_data, target_length=100, timeline=None,
                           target_timeline=None, mode='simple')

        assert len(result) == 100
        assert np.array_equal(result, behavior_data)

    def test_simple_mode_rejects_mismatched_lengths(self):
        """'simple' mode should reject mismatched lengths with appropriate error."""
        behavior_data = np.arange(50)

        with pytest.raises(ValueError) as exc_info:
            align_data(behavior_data, target_length=100, timeline=None,
                      target_timeline=None, mode='simple')

        error_msg = str(exc_info.value).lower()
        assert "simple" in error_msg
        assert "matching lengths" in error_msg


class TestCropMode:
    """Tests for 'crop' alignment mode."""

    def test_crop_mode_handles_equal_lengths(self):
        """'crop' mode should handle equal lengths without cropping."""
        behavior_data = np.arange(100)
        result = align_data(behavior_data, target_length=100, timeline=None,
                           target_timeline=None, mode='crop')

        assert len(result) == 100
        assert np.array_equal(result, behavior_data)

    def test_crop_mode_crops_longer_data(self):
        """'crop' mode should crop longer data correctly."""
        behavior_data = np.arange(150)
        result = align_data(behavior_data, target_length=100, timeline=None,
                           target_timeline=None, mode='crop')

        assert len(result) == 100
        assert np.array_equal(result, behavior_data[:100])

    def test_crop_mode_rejects_shorter_data(self):
        """'crop' mode should reject shorter data with error."""
        behavior_data = np.arange(50)

        with pytest.raises(ValueError) as exc_info:
            align_data(behavior_data, target_length=100, timeline=None,
                      target_timeline=None, mode='crop')

        assert "shorter than calcium" in str(exc_info.value)


class TestFactorMode:
    """Tests for 'factor' alignment mode."""

    def test_factor_mode_not_implemented(self):
        """'factor' mode should raise NotImplementedError."""
        behavior_data = np.arange(100)
        calcium_timeline = np.linspace(0, 10, 150)

        with pytest.raises(NotImplementedError) as exc_info:
            align_data(behavior_data, target_length=150,
                      timeline=None, target_timeline=calcium_timeline,
                      mode='factor')

        assert "factor is not implemented" in str(exc_info.value)


class TestAutomaticModeSelection:
    """Tests for automatic mode selection in align_all_data."""

    def test_error_when_only_source_timeline_missing(self):
        """Should raise error when only source timeline missing (suspicious configuration)."""
        # Behavior with different length and no timeline, but calcium HAS timeline
        # This is suspicious - having only one timeline suggests data problem
        filtered_data = {
            'Calcium': {
                'Calcium': np.random.rand(10, 100),  # 10 neurons, 100 timepoints
                'timeline': np.linspace(0, 10, 100),  # HAS timeline
                'fps': 10
            },
            'Behavior_auto': [
                {
                    'Speed': np.random.rand(50),  # 50 points - different length
                    'timeline': None,  # MISSING timeline
                    'fps': None
                }
            ]
        }

        # Should raise error about suspicious configuration
        with pytest.raises(ValueError) as exc_info:
            align_all_data(filtered_data, force_pathway=None)

        error_msg = str(exc_info.value)
        assert "Only target timeline present" in error_msg
        assert "suspicious" in error_msg.lower()
        assert "force_mode" in error_msg.lower()

    def test_forced_mode_overrides_automatic_selection(self):
        """Forced mode should override automatic selection."""
        # Same data as previous test, but with force_mode
        filtered_data = {
            'Calcium': {
                'Calcium': np.random.rand(10, 100),
                'timeline': np.linspace(0, 10, 100),
                'fps': 10
            },
            'Behavior_auto': [
                {
                    'Speed': np.random.rand(50),
                    'timeline': None,
                    'fps': None
                }
            ]
        }

        # Force cast_to_ca mode should work
        aligned_data, align_log, mode_stats = align_all_data(filtered_data, force_pathway='cast_to_ca')

        # Check that speed was aligned (key might be lowercase)
        feature_keys = {k.lower() for k in aligned_data.keys()}
        assert 'speed' in feature_keys
        assert 'cast_to_ca' in mode_stats
