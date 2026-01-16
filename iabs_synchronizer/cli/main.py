"""
Command-line interface for IABS Data Synchronizer.

Provides CLI commands for synchronizing neuroscience data:
- sync: Synchronize single experiment
- batch: Synchronize multiple experiments
- validate: Validate experiment data
- list: List available experiments
"""

import argparse
import sys
import json

from ..pipeline.synchronizer import Synchronizer
from ..config import SyncConfig
from ..core.postprocessing import print_alignment_summary


def load_rename_dict(rename_file: str) -> dict:
    """
    Load rename dictionary from JSON file.

    Args:
        rename_file: Path to JSON file with rename mapping

    Returns:
        dict: Rename dictionary

    Example JSON format:
        {
            "X": "x_position",
            "Y": "y_position",
            "Speed": "locomotion_speed"
        }
    """
    with open(rename_file, 'r') as f:
        return json.load(f)


def cmd_sync(args):
    """Handle 'sync' command for single experiment."""
    # Load rename dict if provided
    rename_dict = None
    if args.rename:
        rename_dict = load_rename_dict(args.rename)

    # Create synchronizer
    sync = Synchronizer(root_path=args.root)

    # Synchronize experiment
    print(f"Synchronizing experiment: {args.experiment}")
    print(f"Root path: {args.root}")
    if args.mode:
        print(f"Forcing alignment mode: {args.mode}")

    try:
        result = sync.synchronize_experiment(
            args.experiment,
            force_mode=args.mode,
            rename_dict=rename_dict,
            exclude_list=args.exclude
        )

        # Print summary
        print("\n" + "=" * 60)
        print_alignment_summary(result.aligned_data)

        # Save output
        output_path = args.output or f'{args.experiment}_aligned.npz'
        result.save(output_path)
        print(f"\nSaved to: {output_path}")

        # Print logs if verbose
        if args.verbose:
            print("\n" + "=" * 60)
            print("DETAILED LOGS:")
            print(result.get_full_log())

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_batch(args):
    """Handle 'batch' command for multiple experiments."""
    # Load rename dict if provided
    rename_dict = None
    if args.rename:
        rename_dict = load_rename_dict(args.rename)

    # Create synchronizer
    sync = Synchronizer(root_path=args.root)

    # Synchronize batch
    print(f"Synchronizing {len(args.experiments)} experiments")
    print(f"Root path: {args.root}")
    if args.output_dir:
        print(f"Output directory: {args.output_dir}")

    try:
        results = sync.synchronize_batch(
            args.experiments,
            output_dir=args.output_dir,
            force_mode=args.mode,
            rename_dict=rename_dict,
            exclude_list=args.exclude
        )

        print(f"\nSuccessfully synchronized {len(results)} experiments")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_validate(args):
    """Handle 'validate' command for experiment validation."""
    sync = Synchronizer(root_path=args.root)

    print(f"Validating experiment: {args.experiment}")
    print(f"Root path: {args.root}\n")

    try:
        report = sync.validate_experiment(args.experiment)

        print("Validation Report:")
        print("=" * 60)
        print(f"Valid: {report['valid']}")
        print(f"Has Calcium: {report['has_calcium']}")
        print(f"Available data pieces: {', '.join(report['available_pieces']) if report['available_pieces'] else 'None'}")

        if report['errors']:
            print("\nErrors:")
            for error in report['errors']:
                print(f"  - {error}")

        return 0 if report['valid'] else 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_list(args):
    """Handle 'list' command for listing experiments."""
    sync = Synchronizer(root_path=args.root)

    print(f"Listing experiments in: {args.root}\n")

    try:
        experiments = sync.list_experiments()

        if not experiments:
            print("No experiments found")
            return 1

        print(f"Found {len(experiments)} experiments:")
        for exp in experiments:
            print(f"  - {exp}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='iabs-sync',
        description='IABS Data Synchronizer - Align neuroscience recordings to common timeline',
        epilog='For detailed help on a command: iabs-sync <command> --help'
    )

    parser.add_argument('--version', action='version', version='%(prog)s 1.1.0')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # ========== SYNC COMMAND ==========
    sync_parser = subparsers.add_parser(
        'sync',
        help='Synchronize single experiment',
        description='Synchronize all data for a single experiment to common timeline'
    )
    sync_parser.add_argument('experiment', help='Experiment name (subdirectory)')
    sync_parser.add_argument('--root', required=True, help='Root data directory path')
    sync_parser.add_argument('--output', '-o', help='Output .npz file path')
    sync_parser.add_argument(
        '--mode',
        choices=['2 timelines', 'simple', 'cast_to_ca', 'crop'],
        help='Force specific alignment mode (default: auto-select)'
    )
    sync_parser.add_argument(
        '--rename',
        help='JSON file with feature rename mapping'
    )
    sync_parser.add_argument(
        '--exclude',
        nargs='+',
        help='Features to exclude from output'
    )
    sync_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed logs'
    )
    sync_parser.set_defaults(func=cmd_sync)

    # ========== BATCH COMMAND ==========
    batch_parser = subparsers.add_parser(
        'batch',
        help='Synchronize multiple experiments',
        description='Synchronize multiple experiments in batch mode'
    )
    batch_parser.add_argument(
        'experiments',
        nargs='+',
        help='Experiment names to synchronize'
    )
    batch_parser.add_argument('--root', required=True, help='Root data directory path')
    batch_parser.add_argument(
        '--output-dir', '-o',
        help='Output directory for aligned .npz files'
    )
    batch_parser.add_argument(
        '--mode',
        choices=['2 timelines', 'simple', 'cast_to_ca', 'crop'],
        help='Force specific alignment mode for all experiments'
    )
    batch_parser.add_argument(
        '--rename',
        help='JSON file with feature rename mapping (applied to all)'
    )
    batch_parser.add_argument(
        '--exclude',
        nargs='+',
        help='Features to exclude from output (applied to all)'
    )
    batch_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed logs'
    )
    batch_parser.set_defaults(func=cmd_batch)

    # ========== VALIDATE COMMAND ==========
    validate_parser = subparsers.add_parser(
        'validate',
        help='Validate experiment data',
        description='Check if experiment data is valid without synchronizing'
    )
    validate_parser.add_argument('experiment', help='Experiment name to validate')
    validate_parser.add_argument('--root', required=True, help='Root data directory path')
    validate_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed information'
    )
    validate_parser.set_defaults(func=cmd_validate)

    # ========== LIST COMMAND ==========
    list_parser = subparsers.add_parser(
        'list',
        help='List available experiments',
        description='List all experiment directories in root path'
    )
    list_parser.add_argument('--root', required=True, help='Root data directory path')
    list_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed information'
    )
    list_parser.set_defaults(func=cmd_list)

    # Parse arguments
    args = parser.parse_args()

    # Show help if no command provided
    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
