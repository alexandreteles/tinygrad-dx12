"""Refresh the vendored tinygrad specification files from GitHub."""

import shutil
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    """Replace ``tinygrad_spec`` with the current upstream ``spec`` directory.

    The tinygrad repository is shallow-cloned with a sparse checkout so that
    only its ``spec`` directory is downloaded. The checked-out files are then
    copied into the project-local ``tinygrad_spec`` directory. Git must be
    available on ``PATH`` and network access is required.

    Raises:
        subprocess.CalledProcessError: If a Git operation fails.
        FileNotFoundError: If Git is not installed or cannot be found.
    """
    project_root = Path(__file__).resolve().parents[2]
    destination = project_root / "tinygrad_spec"

    with tempfile.TemporaryDirectory(prefix="tinygrad-spec-") as temporary_directory:
        checkout = Path(temporary_directory) / "tinygrad"
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                "--branch",
                "master",
                "https://github.com/tinygrad/tinygrad.git",
                str(checkout),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "sparse-checkout", "set", "spec"],
            cwd=checkout,
            check=True,
        )

        source = checkout / "spec"
        destination.mkdir(parents=True, exist_ok=True)
        for existing in destination.iterdir():
            if existing.is_dir():
                shutil.rmtree(existing)
            else:
                existing.unlink()
        for upstream_file in source.iterdir():
            target = destination / upstream_file.name
            if upstream_file.is_dir():
                shutil.copytree(upstream_file, target)
            else:
                shutil.copy2(upstream_file, target)

        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=checkout,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    print(f"Updated {destination} from tinygrad commit {commit}.")


if __name__ == "__main__":
    main()
