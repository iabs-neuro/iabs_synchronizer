"""
Test "2 timelines" mode with sparse/irregular sampling patterns.

Testing scenarios where timestamps are valid but sparsely sampled,
and where sampling patterns don't align between behavior and calcium.
"""

import numpy as np
import pytest
from iabs_synchronizer.core.alignment import align_data


class TestSparseSampling:
    """Tests for sparse sampling patterns."""

    def test_behavior_sparse_calcium_dense(self):
        """Test when behavior is sparsely sampled, calcium is dense."""
        # Behavior: sampled only at specific times [0,1,2,3,4, 10,11,12,13,14]
        # Missing: 5,6,7,8,9
        behavior_timeline = np.array([0., 1., 2., 3., 4., 10., 11., 12., 13., 14.])
        behavior_data = np.array([100., 110., 120., 130., 140., 200., 210., 220., 230., 240.])

        # Calcium: continuous dense sampling [0,1,2,...,14]
        calcium_timeline = np.linspace(0, 14, 15)

        result = align_data(
            behavior_data,
            target_length=len(calcium_timeline),
            timeline=behavior_timeline,
            target_timeline=calcium_timeline,
            mode='2 timelines'
        )

        assert len(result) == len(calcium_timeline)

        # Linear interpolation should fill values for t=5-9s
        # This is expected behavior for sparse sampling

    def test_calcium_sparse_behavior_dense(self):
        """Test when calcium is sparsely sampled, behavior is dense."""
        # Behavior: dense [0,1,2,...,14]
        behavior_timeline = np.linspace(0, 14, 15)
        behavior_data = behavior_timeline * 10  # Simple linear data

        # Calcium: sparse [0,1,2,3,4, 10,11,12,13,14]
        calcium_timeline = np.array([0., 1., 2., 3., 4., 10., 11., 12., 13., 14.])

        result = align_data(
            behavior_data,
            target_length=len(calcium_timeline),
            timeline=behavior_timeline,
            target_timeline=calcium_timeline,
            mode='2 timelines'
        )

        assert len(result) == len(calcium_timeline)

        # Calcium only asks for values at its sparse timestamps
        # Behavior data exists at all these points


class TestNonOverlappingSampling:
    """Tests for non-overlapping sparse patterns."""

    def test_non_overlapping_sparse_patterns(self):
        """Test when behavior and calcium sample at different times."""
        # Behavior: even timestamps [0,2,4,6,8,10,12,14]
        behavior_timeline = np.array([0., 2., 4., 6., 8., 10., 12., 14.])
        behavior_data = behavior_timeline * 10

        # Calcium: odd timestamps [1,3,5,7,9,11,13]
        calcium_timeline = np.array([1., 3., 5., 7., 9., 11., 13.])

        result = align_data(
            behavior_data,
            target_length=len(calcium_timeline),
            timeline=behavior_timeline,
            target_timeline=calcium_timeline,
            mode='2 timelines'
        )

        assert len(result) == len(calcium_timeline)

        # Every calcium timestamp falls between behavior samples
        # Linear interpolation provides estimates


class TestVerySparseSampling:
    """Tests for very sparse sampling with large gaps."""

    def test_very_sparse_with_large_gaps(self):
        """Test very sparse sampling over long time period."""
        # Behavior: only 3 samples over 0-20s
        behavior_timeline = np.array([0., 10., 20.])
        behavior_data = np.array([100., 200., 300.])

        # Calcium: wants many samples
        calcium_timeline = np.linspace(0, 20, 21)  # Every 1s

        result = align_data(
            behavior_data,
            target_length=len(calcium_timeline),
            timeline=behavior_timeline,
            target_timeline=calcium_timeline,
            mode='2 timelines'
        )

        assert len(result) == len(calcium_timeline)

        # Linear interpolation fills all gaps
        # Assumes signal changes linearly during unsampled periods


class TestMisalignedSparsePatterns:
    """Tests for misaligned sparse patterns (realistic case)."""

    def test_misaligned_sparse_patterns(self):
        """Test realistic irregular sampling patterns."""
        # Behavior: irregular sampling [0, 1, 3, 4, 7, 9, 10, 13, 15]
        behavior_timeline = np.array([0., 1., 3., 4., 7., 9., 10., 13., 15.])
        behavior_data = np.sin(behavior_timeline)  # Some signal

        # Calcium: different irregular sampling [0, 2, 3, 5, 8, 10, 12, 15]
        calcium_timeline = np.array([0., 2., 3., 5., 8., 10., 12., 15.])

        result = align_data(
            behavior_data,
            target_length=len(calcium_timeline),
            timeline=behavior_timeline,
            target_timeline=calcium_timeline,
            mode='2 timelines'
        )

        assert len(result) == len(calcium_timeline)

        # Both are irregularly sampled, some timestamps align, some don't
        # Alignment by timestamp overlap works correctly
