"""
Comprehensive test suite for CLI commands.

Tests all CLI commands (sync, batch, validate, list) with various scenarios,
error handling, and argument validation.
"""

import pytest
import sys
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock, call
from io import StringIO

from iabs_synchronizer.cli.main import (
    main, cmd_sync, cmd_batch, cmd_validate, cmd_list, load_rename_dict
)
from iabs_synchronizer.models.data_structures import SyncResult


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_synchronizer():
    """Create a mock Synchronizer instance."""
    with patch('iabs_synchronizer.cli.main.Synchronizer') as MockSync:
        mock_instance = MockSync.return_value

        # Mock sync result
        mock_result = Mock(spec=SyncResult)
        mock_result.aligned_data = {
            'Calcium': {'neuron_0': [1, 2, 3]},
            'Behavior_auto': {'feature_1': [1, 2, 3]}
        }
        mock_result.get_full_log.return_value = "Test log output"
        mock_result.save = Mock()

        mock_instance.synchronize_experiment.return_value = mock_result
        mock_instance.synchronize_batch.return_value = [mock_result, mock_result]
        mock_instance.validate_experiment.return_value = {
            'valid': True,
            'has_calcium': True,
            'available_pieces': ['Calcium', 'Behavior_auto'],
            'errors': []
        }
        mock_instance.list_experiments.return_value = ['exp1', 'exp2', 'exp3']

        yield mock_instance


@pytest.fixture
def temp_rename_file():
    """Create a temporary rename JSON file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        rename_dict = {
            'X': 'x_position',
            'Y': 'y_position',
            'Speed': 'locomotion_speed'
        }
        json.dump(rename_dict, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def captured_output():
    """Context manager for capturing stdout and stderr."""
    from contextlib import contextmanager

    @contextmanager
    def _capture():
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        try:
            yield sys.stdout, sys.stderr
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    return _capture


# ============================================================================
# LOAD RENAME DICT TESTS
# ============================================================================

class TestLoadRenameDict:
    """Test load_rename_dict helper function."""

    def test_load_valid_json(self, temp_rename_file):
        """Test loading valid rename JSON."""
        result = load_rename_dict(temp_rename_file)

        assert isinstance(result, dict)
        assert result['X'] == 'x_position'
        assert result['Y'] == 'y_position'
        assert result['Speed'] == 'locomotion_speed'

    def test_load_missing_file(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_rename_dict('/nonexistent/file.json')

    def test_load_invalid_json(self):
        """Test loading invalid JSON raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content {")
            temp_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                load_rename_dict(temp_path)
        finally:
            os.remove(temp_path)


# ============================================================================
# SYNC COMMAND TESTS
# ============================================================================

class TestCmdSync:
    """Test 'sync' command functionality."""

    def test_sync_basic(self, mock_synchronizer):
        """Test basic sync command with required arguments."""
        args = Mock()
        args.experiment = 'test_exp'
        args.root = '/data'
        args.output = None
        args.mode = None
        args.rename = None
        args.exclude = None
        args.verbose = False

        with patch('iabs_synchronizer.cli.main.print_alignment_summary'):
            exit_code = cmd_sync(args)

        assert exit_code == 0
        mock_synchronizer.synchronize_experiment.assert_called_once_with(
            'test_exp',
            force_mode=None,
            rename_dict=None,
            exclude_list=None
        )

    def test_sync_with_output_path(self, mock_synchronizer):
        """Test sync command with custom output path."""
        args = Mock()
        args.experiment = 'test_exp'
        args.root = '/data'
        args.output = 'custom_output.npz'
        args.mode = None
        args.rename = None
        args.exclude = None
        args.verbose = False

        mock_result = mock_synchronizer.synchronize_experiment.return_value

        with patch('iabs_synchronizer.cli.main.print_alignment_summary'):
            exit_code = cmd_sync(args)

        assert exit_code == 0
        mock_result.save.assert_called_once_with('custom_output.npz')

    def test_sync_with_forced_mode(self, mock_synchronizer):
        """Test sync command with forced alignment mode."""
        args = Mock()
        args.experiment = 'test_exp'
        args.root = '/data'
        args.output = None
        args.mode = '2 timelines'
        args.rename = None
        args.exclude = None
        args.verbose = False

        with patch('iabs_synchronizer.cli.main.print_alignment_summary'):
            exit_code = cmd_sync(args)

        assert exit_code == 0
        mock_synchronizer.synchronize_experiment.assert_called_once_with(
            'test_exp',
            force_mode='2 timelines',
            rename_dict=None,
            exclude_list=None
        )

    def test_sync_with_rename_dict(self, mock_synchronizer, temp_rename_file):
        """Test sync command with rename dictionary."""
        args = Mock()
        args.experiment = 'test_exp'
        args.root = '/data'
        args.output = None
        args.mode = None
        args.rename = temp_rename_file
        args.exclude = None
        args.verbose = False

        with patch('iabs_synchronizer.cli.main.print_alignment_summary'):
            exit_code = cmd_sync(args)

        assert exit_code == 0
        call_args = mock_synchronizer.synchronize_experiment.call_args
        assert call_args[1]['rename_dict'] is not None
        assert call_args[1]['rename_dict']['X'] == 'x_position'

    def test_sync_with_exclude_list(self, mock_synchronizer):
        """Test sync command with feature exclusion."""
        args = Mock()
        args.experiment = 'test_exp'
        args.root = '/data'
        args.output = None
        args.mode = None
        args.rename = None
        args.exclude = ['feature1', 'feature2']
        args.verbose = False

        with patch('iabs_synchronizer.cli.main.print_alignment_summary'):
            exit_code = cmd_sync(args)

        assert exit_code == 0
        mock_synchronizer.synchronize_experiment.assert_called_once_with(
            'test_exp',
            force_mode=None,
            rename_dict=None,
            exclude_list=['feature1', 'feature2']
        )

    def test_sync_verbose_prints_logs(self, mock_synchronizer):
        """Test sync command in verbose mode prints detailed logs."""
        args = Mock()
        args.experiment = 'test_exp'
        args.root = '/data'
        args.output = None
        args.mode = None
        args.rename = None
        args.exclude = None
        args.verbose = True

        mock_result = mock_synchronizer.synchronize_experiment.return_value

        with patch('iabs_synchronizer.cli.main.print_alignment_summary'):
            exit_code = cmd_sync(args)

        assert exit_code == 0
        mock_result.get_full_log.assert_called_once()

    def test_sync_handles_exception(self, mock_synchronizer, captured_output):
        """Test sync command handles exceptions gracefully."""
        args = Mock()
        args.experiment = 'test_exp'
        args.root = '/data'
        args.output = None
        args.mode = None
        args.rename = None
        args.exclude = None
        args.verbose = False

        mock_synchronizer.synchronize_experiment.side_effect = ValueError("Test error")

        with captured_output() as (stdout, stderr):
            exit_code = cmd_sync(args)

        assert exit_code == 1
        assert "Error: Test error" in stderr.getvalue()

    def test_sync_exception_with_verbose(self, mock_synchronizer):
        """Test sync command prints traceback in verbose mode."""
        args = Mock()
        args.experiment = 'test_exp'
        args.root = '/data'
        args.output = None
        args.mode = None
        args.rename = None
        args.exclude = None
        args.verbose = True

        mock_synchronizer.synchronize_experiment.side_effect = ValueError("Test error")

        with patch('traceback.print_exc') as mock_traceback:
            exit_code = cmd_sync(args)

        assert exit_code == 1
        mock_traceback.assert_called_once()


# ============================================================================
# BATCH COMMAND TESTS
# ============================================================================

class TestCmdBatch:
    """Test 'batch' command functionality."""

    def test_batch_basic(self, mock_synchronizer):
        """Test basic batch command with multiple experiments."""
        args = Mock()
        args.experiments = ['exp1', 'exp2', 'exp3']
        args.root = '/data'
        args.output_dir = None
        args.mode = None
        args.rename = None
        args.exclude = None
        args.verbose = False

        exit_code = cmd_batch(args)

        assert exit_code == 0
        mock_synchronizer.synchronize_batch.assert_called_once_with(
            ['exp1', 'exp2', 'exp3'],
            output_dir=None,
            force_mode=None,
            rename_dict=None,
            exclude_list=None
        )

    def test_batch_with_output_dir(self, mock_synchronizer):
        """Test batch command with custom output directory."""
        args = Mock()
        args.experiments = ['exp1', 'exp2']
        args.root = '/data'
        args.output_dir = '/output'
        args.mode = None
        args.rename = None
        args.exclude = None
        args.verbose = False

        exit_code = cmd_batch(args)

        assert exit_code == 0
        mock_synchronizer.synchronize_batch.assert_called_once_with(
            ['exp1', 'exp2'],
            output_dir='/output',
            force_mode=None,
            rename_dict=None,
            exclude_list=None
        )

    def test_batch_with_forced_mode(self, mock_synchronizer):
        """Test batch command with forced alignment mode."""
        args = Mock()
        args.experiments = ['exp1']
        args.root = '/data'
        args.output_dir = None
        args.mode = 'crop'
        args.rename = None
        args.exclude = None
        args.verbose = False

        exit_code = cmd_batch(args)

        assert exit_code == 0
        call_args = mock_synchronizer.synchronize_batch.call_args
        assert call_args[1]['force_mode'] == 'crop'

    def test_batch_with_rename_dict(self, mock_synchronizer, temp_rename_file):
        """Test batch command with rename dictionary."""
        args = Mock()
        args.experiments = ['exp1']
        args.root = '/data'
        args.output_dir = None
        args.mode = None
        args.rename = temp_rename_file
        args.exclude = None
        args.verbose = False

        exit_code = cmd_batch(args)

        assert exit_code == 0
        call_args = mock_synchronizer.synchronize_batch.call_args
        assert call_args[1]['rename_dict'] is not None

    def test_batch_handles_exception(self, mock_synchronizer, captured_output):
        """Test batch command handles exceptions gracefully."""
        args = Mock()
        args.experiments = ['exp1']
        args.root = '/data'
        args.output_dir = None
        args.mode = None
        args.rename = None
        args.exclude = None
        args.verbose = False

        mock_synchronizer.synchronize_batch.side_effect = FileNotFoundError("Data not found")

        with captured_output() as (stdout, stderr):
            exit_code = cmd_batch(args)

        assert exit_code == 1
        assert "Error: Data not found" in stderr.getvalue()


# ============================================================================
# VALIDATE COMMAND TESTS
# ============================================================================

class TestCmdValidate:
    """Test 'validate' command functionality."""

    def test_validate_valid_experiment(self, mock_synchronizer):
        """Test validate command with valid experiment."""
        args = Mock()
        args.experiment = 'test_exp'
        args.root = '/data'
        args.verbose = False

        exit_code = cmd_validate(args)

        assert exit_code == 0
        mock_synchronizer.validate_experiment.assert_called_once_with('test_exp')

    def test_validate_invalid_experiment(self, mock_synchronizer, captured_output):
        """Test validate command with invalid experiment."""
        args = Mock()
        args.experiment = 'test_exp'
        args.root = '/data'
        args.verbose = False

        mock_synchronizer.validate_experiment.return_value = {
            'valid': False,
            'has_calcium': False,
            'available_pieces': [],
            'errors': ['Calcium data missing']
        }

        with captured_output() as (stdout, stderr):
            exit_code = cmd_validate(args)

        assert exit_code == 1
        output = stdout.getvalue()
        assert 'Valid: False' in output
        assert 'Calcium data missing' in output

    def test_validate_with_errors(self, mock_synchronizer, captured_output):
        """Test validate command displays errors."""
        args = Mock()
        args.experiment = 'test_exp'
        args.root = '/data'
        args.verbose = False

        mock_synchronizer.validate_experiment.return_value = {
            'valid': False,
            'has_calcium': True,
            'available_pieces': ['Calcium'],
            'errors': ['Error 1', 'Error 2']
        }

        with captured_output() as (stdout, stderr):
            exit_code = cmd_validate(args)

        assert exit_code == 1
        output = stdout.getvalue()
        assert 'Error 1' in output
        assert 'Error 2' in output

    def test_validate_handles_exception(self, mock_synchronizer, captured_output):
        """Test validate command handles exceptions gracefully."""
        args = Mock()
        args.experiment = 'test_exp'
        args.root = '/data'
        args.verbose = False

        mock_synchronizer.validate_experiment.side_effect = RuntimeError("Validation error")

        with captured_output() as (stdout, stderr):
            exit_code = cmd_validate(args)

        assert exit_code == 1
        assert "Error: Validation error" in stderr.getvalue()


# ============================================================================
# LIST COMMAND TESTS
# ============================================================================

class TestCmdList:
    """Test 'list' command functionality."""

    def test_list_experiments(self, mock_synchronizer, captured_output):
        """Test list command with experiments present."""
        args = Mock()
        args.root = '/data'
        args.verbose = False

        with captured_output() as (stdout, stderr):
            exit_code = cmd_list(args)

        assert exit_code == 0
        mock_synchronizer.list_experiments.assert_called_once()
        output = stdout.getvalue()
        assert 'Found 3 experiments' in output
        assert 'exp1' in output
        assert 'exp2' in output
        assert 'exp3' in output

    def test_list_no_experiments(self, mock_synchronizer, captured_output):
        """Test list command with no experiments found."""
        args = Mock()
        args.root = '/data'
        args.verbose = False

        mock_synchronizer.list_experiments.return_value = []

        with captured_output() as (stdout, stderr):
            exit_code = cmd_list(args)

        assert exit_code == 1
        output = stdout.getvalue()
        assert 'No experiments found' in output

    def test_list_handles_exception(self, mock_synchronizer, captured_output):
        """Test list command handles exceptions gracefully."""
        args = Mock()
        args.root = '/data'
        args.verbose = False

        mock_synchronizer.list_experiments.side_effect = PermissionError("Access denied")

        with captured_output() as (stdout, stderr):
            exit_code = cmd_list(args)

        assert exit_code == 1
        assert "Error: Access denied" in stderr.getvalue()


# ============================================================================
# MAIN FUNCTION TESTS
# ============================================================================

class TestMain:
    """Test main CLI entry point."""

    def test_main_no_command_shows_help(self):
        """Test main with no command shows help."""
        with patch('sys.argv', ['iabs-sync']):
            with patch('argparse.ArgumentParser.print_help') as mock_help:
                exit_code = main()

        assert exit_code == 1
        mock_help.assert_called_once()

    def test_main_sync_command(self, mock_synchronizer):
        """Test main executes sync command."""
        with patch('sys.argv', ['iabs-sync', 'sync', 'test_exp', '--root', '/data']):
            with patch('iabs_synchronizer.cli.main.print_alignment_summary'):
                exit_code = main()

        assert exit_code == 0
        mock_synchronizer.synchronize_experiment.assert_called_once()

    def test_main_batch_command(self, mock_synchronizer):
        """Test main executes batch command."""
        with patch('sys.argv', ['iabs-sync', 'batch', 'exp1', 'exp2', '--root', '/data']):
            exit_code = main()

        assert exit_code == 0
        mock_synchronizer.synchronize_batch.assert_called_once()

    def test_main_validate_command(self, mock_synchronizer):
        """Test main executes validate command."""
        with patch('sys.argv', ['iabs-sync', 'validate', 'test_exp', '--root', '/data']):
            exit_code = main()

        assert exit_code == 0
        mock_synchronizer.validate_experiment.assert_called_once()

    def test_main_list_command(self, mock_synchronizer):
        """Test main executes list command."""
        with patch('sys.argv', ['iabs-sync', 'list', '--root', '/data']):
            exit_code = main()

        assert exit_code == 0
        mock_synchronizer.list_experiments.assert_called_once()

    def test_main_version_flag(self):
        """Test main with --version flag."""
        with patch('sys.argv', ['iabs-sync', '--version']):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 0


# ============================================================================
# ARGUMENT VALIDATION TESTS
# ============================================================================

class TestArgumentValidation:
    """Test CLI argument validation."""

    def test_sync_missing_root(self):
        """Test sync command rejects missing --root."""
        with patch('sys.argv', ['iabs-sync', 'sync', 'test_exp']):
            with pytest.raises(SystemExit):
                main()

    def test_batch_missing_experiments(self):
        """Test batch command rejects no experiments."""
        with patch('sys.argv', ['iabs-sync', 'batch', '--root', '/data']):
            with pytest.raises(SystemExit):
                main()

    def test_validate_missing_experiment(self):
        """Test validate command rejects missing experiment name."""
        with patch('sys.argv', ['iabs-sync', 'validate', '--root', '/data']):
            with pytest.raises(SystemExit):
                main()

    def test_invalid_mode_choice(self):
        """Test invalid alignment mode is rejected."""
        with patch('sys.argv', ['iabs-sync', 'sync', 'test_exp', '--root', '/data', '--mode', 'invalid']):
            with pytest.raises(SystemExit):
                main()

    def test_valid_mode_choices(self, mock_synchronizer):
        """Test all valid alignment modes are accepted."""
        valid_modes = ['2 timelines', 'simple', 'cast_to_ca', 'crop']

        for mode in valid_modes:
            with patch('sys.argv', ['iabs-sync', 'sync', 'test_exp', '--root', '/data', '--mode', mode]):
                with patch('iabs_synchronizer.cli.main.print_alignment_summary'):
                    exit_code = main()

            assert exit_code == 0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestCLIIntegration:
    """Integration tests for CLI workflow."""

    def test_sync_full_workflow(self, mock_synchronizer, temp_rename_file):
        """Test complete sync workflow with all options."""
        argv = [
            'iabs-sync', 'sync', 'test_exp',
            '--root', '/data',
            '--output', 'output.npz',
            '--mode', '2 timelines',
            '--rename', temp_rename_file,
            '--exclude', 'feature1', 'feature2',
            '--verbose'
        ]

        with patch('sys.argv', argv):
            with patch('iabs_synchronizer.cli.main.print_alignment_summary'):
                exit_code = main()

        assert exit_code == 0

        # Verify all arguments were passed correctly
        call_args = mock_synchronizer.synchronize_experiment.call_args
        assert call_args[0][0] == 'test_exp'
        assert call_args[1]['force_mode'] == '2 timelines'
        assert call_args[1]['rename_dict'] is not None
        assert call_args[1]['exclude_list'] == ['feature1', 'feature2']

    def test_batch_multiple_experiments(self, mock_synchronizer):
        """Test batch processing of multiple experiments."""
        argv = [
            'iabs-sync', 'batch',
            'exp1', 'exp2', 'exp3', 'exp4',
            '--root', '/data',
            '--output-dir', '/output',
            '--verbose'
        ]

        with patch('sys.argv', argv):
            exit_code = main()

        assert exit_code == 0

        call_args = mock_synchronizer.synchronize_batch.call_args
        assert len(call_args[0][0]) == 4
        assert call_args[1]['output_dir'] == '/output'
