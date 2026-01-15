"""
Comprehensive tests for gap detection functionality.
"""

import numpy as np
import pytest
from iabs_synchronizer.utils.gap_detection import (
    detect_timeline_gaps,
    print_gap_report,
    check_timeline_quality,
    calculate_robust_fps
)


class TestBasicGapDetection:
    """Tests for basic gap detection functionality."""

    def test_perfect_regular_sampling_no_gaps(self):
        """Perfect regular sampling @ 10fps should detect no gaps."""
        timeline = np.linspace(0, 10, 101)  # 0 to 10s, 101 points = 10fps
        has_gaps, info = detect_timeline_gaps(timeline)

        assert has_gaps is False
        assert info['n_gaps'] == 0
        assert abs(info['estimated_fps'] - 10.0) < 0.1
        assert info['data_coverage'] == 100.0

    def test_single_large_gap(self):
        """Single large gap in middle should be detected."""
        # Regular sampling 0-3s, gap, then 10-13s
        part1 = np.linspace(0, 3, 31)  # 0.1s intervals
        part2 = np.linspace(10, 13, 31)  # 0.1s intervals
        timeline = np.concatenate([part1, part2])

        has_gaps, info = detect_timeline_gaps(timeline)

        assert has_gaps is True
        assert info['n_gaps'] == 1
        assert abs(info['gap_durations'][0] - 7.0) < 0.1
        assert abs(info['estimated_fps'] - 10.0) < 0.5

    def test_multiple_gaps(self):
        """Multiple gaps should all be detected."""
        # Three segments with gaps between
        seg1 = np.linspace(0, 2, 21)  # 0-2s @ 10fps
        seg2 = np.linspace(5, 7, 21)  # 5-7s @ 10fps (gap 2-5)
        seg3 = np.linspace(10, 12, 21)  # 10-12s @ 10fps (gap 7-10)
        timeline = np.concatenate([seg1, seg2, seg3])

        has_gaps, info = detect_timeline_gaps(timeline)

        assert has_gaps is True
        assert info['n_gaps'] == 2
        assert abs(sum(info['gap_durations']) - 6.0) < 0.1


class TestNoisyData:
    """Tests for handling noisy sampling."""

    def test_noisy_sampling_no_false_positives(self):
        """Noisy sampling should NOT detect gaps."""
        # Regular sampling with small noise
        timeline = np.linspace(0, 10, 101)
        noise = np.random.normal(0, 0.005, 101)  # Small noise (<5% of interval)
        timeline = timeline + noise
        timeline = np.sort(timeline)  # Ensure sorted

        has_gaps, info = detect_timeline_gaps(timeline, gap_threshold_multiplier=3.0)

        # Should not detect gaps (noise is within threshold)
        assert has_gaps is False

    def test_outliers_do_not_affect_fps_estimation(self):
        """Outliers should not affect FPS estimation (robust statistics)."""
        # Mostly regular @ 20fps, but with a few outliers
        timeline = np.linspace(0, 5, 101)  # 20fps
        # Add a couple outlier intervals (not full gaps)
        timeline[50] = timeline[49] + 0.1  # Slightly larger interval
        timeline[51:] += 0.05  # Shift rest slightly

        has_gaps, info = detect_timeline_gaps(timeline, gap_threshold_multiplier=3.0)

        # Should estimate FPS close to 20 despite outliers
        assert abs(info['estimated_fps'] - 20.0) < 1.0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_short_timeline(self):
        """Very short timeline should be handled gracefully."""
        timeline = np.array([0.0, 0.1, 0.2])

        has_gaps, info = detect_timeline_gaps(timeline)

        # Should complete without error and include warnings
        assert 'warnings' in info


class TestGapThresholdSensitivity:
    """Tests for gap threshold parameter."""

    def test_threshold_affects_detection(self):
        """Gap threshold should affect detection sensitivity."""
        # Timeline with moderate irregularity
        timeline = np.array([0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7])  # 0.2s gap at 0.3-0.5

        # Strict threshold (2x)
        has_gaps_strict, info_strict = detect_timeline_gaps(timeline, gap_threshold_multiplier=2.0)

        # Lenient threshold (5x)
        has_gaps_lenient, info_lenient = detect_timeline_gaps(timeline, gap_threshold_multiplier=5.0)

        # Strict should detect more gaps than lenient
        assert info_strict['n_gaps'] >= info_lenient['n_gaps']


class TestRobustFPS:
    """Tests for robust FPS calculation."""

    def test_robust_fps_with_gaps(self):
        """Robust FPS calculation should work even with gaps."""
        # 30fps data with gap
        part1 = np.arange(0, 3, 1/30)  # 30fps for 3s
        part2 = np.arange(10, 13, 1/30)  # Gap, then 30fps for 3s
        timeline = np.concatenate([part1, part2])

        fps, confident = calculate_robust_fps(timeline, allowed_fps=[10, 20, 30, 40])

        assert abs(fps - 30.0) < 1.0
        assert confident is True


class TestGapReporting:
    """Tests for gap reporting functions."""

    def test_print_gap_report(self, capsys):
        """Gap report should be generated without errors."""
        # Create timeline with known gap
        part1 = np.linspace(0, 5, 51)
        part2 = np.linspace(8, 13, 51)
        timeline = np.concatenate([part1, part2])

        has_gaps, info = detect_timeline_gaps(timeline)

        # Should not raise exception
        print_gap_report(timeline, info, feature_name="Test Feature")

        # Check output was generated
        captured = capsys.readouterr()
        assert "Test Feature" in captured.out


class TestTimelineQuality:
    """Tests for timeline quality checking."""

    def test_timeline_quality_with_warnings(self, capsys):
        """Timeline quality check should produce warnings for gaps."""
        # Timeline with gap
        timeline = np.concatenate([
            np.linspace(0, 2, 21),
            np.linspace(5, 7, 21)
        ])

        gap_info = check_timeline_quality(timeline, feature_name="TestFeature", warn=True)

        assert gap_info['n_gaps'] > 0

        # Check warning was printed
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_timeline_quality_raises_on_gaps(self):
        """Timeline quality check should raise error when configured."""
        # Timeline with gap
        timeline = np.concatenate([
            np.linspace(0, 2, 21),
            np.linspace(5, 7, 21)
        ])

        with pytest.raises(ValueError):
            check_timeline_quality(timeline, feature_name="TestFeature",
                                  warn=False, raise_on_gaps=True)


class TestRealWorldScenarios:
    """Tests for realistic recording scenarios."""

    def test_recording_with_pause(self):
        """Simulate mouse position tracking with recording pause."""
        # Simulate: mouse position tracking @ 40fps, paused for 2 minutes mid-experiment
        fps = 40
        recording1 = np.arange(0, 300, 1/fps)  # 5 minutes
        recording2 = np.arange(420, 600, 1/fps)  # Pause 120s, then 3 more minutes
        timeline = np.concatenate([recording1, recording2])

        has_gaps, info = detect_timeline_gaps(timeline)

        assert has_gaps is True
        assert info['n_gaps'] == 1
        assert 119 < info['gap_durations'][0] < 121  # ~120s gap
