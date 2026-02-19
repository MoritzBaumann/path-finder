from pathlib import Path

import yaml


def load_latest_history(
    output_dir: Path, pattern: str = "*history*.yaml"
) -> tuple[dict, str]:
    """Load most recent training history and return history and filename."""
    history_files = sorted(output_dir.glob(pattern))

    if not history_files:
        raise FileNotFoundError(
            f"No history files found in {output_dir} matching pattern '{pattern}'"
        )

    # Get the most recent file (sorted by filename which includes timestamp)
    latest_file = history_files[-1]

    print(f"\nLoading history from: {latest_file.name}")

    with open(latest_file, "r") as f:
        history = yaml.safe_load(f)

    return (history, latest_file.name)
