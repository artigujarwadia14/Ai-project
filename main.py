"""
Crowd Rush Prediction & Early Warning System
Unified Command-Line Interface (CLI) Orchestrator

Usage Examples:
  python main.py --inspect           # Audit all datasets and annotations
  python main.py --dataset-guide     # Print official download instructions for full datasets
  python main.py --download-nwpu     # Download 10-image official NWPU-Crowd validation subset
  python main.py --evaluate          # Run optical flow, video telemetry, risk timeline & summary CSV
  python main.py --evaluate --full   # Re-run full CV/ML detection, counting, tracking & feature benchmarks
  python main.py --dashboard         # Launch the interactive Streamlit web dashboard
"""

import argparse
import subprocess
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.data_preprocessing.dataset_inspector import DatasetInspector
from src.data_preprocessing.download_real_subset import (
    download_nwpu_real_subset,
    print_full_dataset_instructions,
)
from src.evaluation.benchmark_runner import run_all_benchmarks


def launch_dashboard(port: int = 8501):
    """Launches the Streamlit dashboard using the active Python interpreter."""
    cmd = [sys.executable, "-m", "streamlit", "run", "dashboard/app.py", "--server.port", str(port)]
    print(f"Launching Streamlit dashboard on http://localhost:{port} ...")
    subprocess.run(cmd, check=False)


def main():
    parser = argparse.ArgumentParser(description="Crowd Rush Prediction & Early Warning System CLI")
    parser.add_argument("--inspect", action="store_true", help="Inspect all datasets and print summary report.")
    parser.add_argument("--setup-samples", action="store_true", help="Generate schema-compliant sample datasets.")
    parser.add_argument("--download-nwpu", action="store_true", help="Download 10-image official NWPU-Crowd validation subset.")
    parser.add_argument("--dataset-guide", action="store_true", help="Print full official dataset access instructions.")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation suite and consolidate metrics.")
    parser.add_argument("--full", action="store_true", help="Used with --evaluate to recompute heavy CV detection/counting/tracking.")
    parser.add_argument("--dashboard", action="store_true", help="Launch the Streamlit dashboard.")
    parser.add_argument("--port", type=int, default=8501, help="Port for Streamlit dashboard (default 8501).")

    args = parser.parse_args()

    if not any([args.inspect, args.setup_samples, args.download_nwpu, args.dataset_guide, args.evaluate, args.dashboard]):
        parser.print_help()
        return

    if args.dataset_guide:
        print_full_dataset_instructions()

    if args.setup_samples:
        from src.data_preprocessing.dataset_setup import setup_all_datasets
        setup_all_datasets()

    if args.download_nwpu:
        download_nwpu_real_subset()

    if args.inspect:
        inspector = DatasetInspector()
        print(inspector.generate_markdown_report())

    if args.evaluate:
        run_all_benchmarks(recompute_heavy=args.full)

    if args.dashboard:
        launch_dashboard(port=args.port)


if __name__ == "__main__":
    main()
