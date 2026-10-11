# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 10:47:56 2026

@author: User
"""


"""
Download and organize deepfake research datasets.

Important:
- Use only official or authorized download links.
- FaceForensics++, FakeAVCeleb and DeeperForensics may require approval.
- DFDC requires Kaggle access and acceptance of competition rules.
- Large datasets may require hundreds of GB of free disk space.
"""

from pathlib import Path
from urllib.request import Request, urlopen
import subprocess
import zipfile
import tarfile
import shutil
import csv
import time

# Change this path to your preferred storage drive.
ROOT = Path(r"D:\Deepfake_Datasets")
ROOT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "FaceForensics++": {
        "url": "",  # Paste your authorized direct download URL here
        "archive": "faceforensics.zip",
    },
    "DFDC": {
        "url": "",  # Optional direct URL; otherwise use Kaggle below
        "archive": "dfdc.zip",
    },
    "Celeb-DF-v2": {
        "url": "",  # Paste your authorized direct download URL here
        "archive": "celeb_df_v2.zip",
    },
    "FakeAVCeleb": {
        "url": "",  # Paste your authorized direct download URL here
        "archive": "fakeavceleb.zip",
    },
    "DeeperForensics-1.0": {
        "url": "",  # Paste your authorized direct download URL here
        "archive": "deeperforensics.tar",
    },
    "ForgeryNet": {
        "url": "",  # Paste your authorized direct download URL here
        "archive": "forgerynet.tar",
    },
}


def download_file(url, destination):
    """Download a file with progress and retry support."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    for attempt in range(3):
        try:
            request = Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"}
            )

            with urlopen(request, timeout=60) as response:
                total = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                last_report = time.time()

                with open(partial, "wb") as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break

                        output.write(chunk)
                        downloaded += len(chunk)

                        if time.time() - last_report >= 2:
                            if total:
                                percent = downloaded * 100 / total
                                print(
                                    f"\r{destination.name}: "
                                    f"{percent:.1f}%"
                                )
                            else:
                                print(
                                    f"\r{destination.name}: "
                                    f"{downloaded / (1024**3):.2f} GB"
                                )
                            last_report = time.time()

            partial.replace(destination)
            print(f"\nDownloaded: {destination}")
            return True

        except Exception as error:
            print(f"\nAttempt {attempt + 1}/3 failed: {error}")
            if attempt < 2:
                time.sleep(3)

    print(f"FAILED: {destination}")
    return False


def extract_archive(archive, output_dir):
    """Extract ZIP, TAR, TAR.GZ and TGZ archives safely."""
    archive = Path(archive)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root_resolved = output_dir.resolve()

    def safe_target(name):
        target = (output_dir / name).resolve()
        if target != root_resolved and root_resolved not in target.parents:
            raise ValueError(f"Unsafe archive path: {name}")
        return target

    try:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zf:
                for member in zf.infolist():
                    safe_target(member.filename)
                zf.extractall(output_dir)
            print(f"Extracted ZIP into {output_dir}")

        elif tarfile.is_tarfile(archive):
            with tarfile.open(archive, "r:*") as tf:
                for member in tf.getmembers():
                    safe_target(member.name)
                    if member.issym() or member.islnk():
                        raise ValueError(
                            f"Links are not extracted: {member.name}"
                        )
                tf.extractall(output_dir)
            print(f"Extracted TAR into {output_dir}")

        else:
            print(f"Unknown archive format: {archive}")

    except Exception as error:
        print(f"Extraction failed for {archive}: {error}")


def download_from_config(name, config):
    folder = ROOT / name
    folder.mkdir(parents=True, exist_ok=True)

    url = config["url"].strip()
    if not url:
        print(
            f"\nSKIPPED {name}: no authorized direct URL configured.\n"
            "Obtain access from the dataset's official provider first."
        )
        return

    archive = folder / config["archive"]

    if not archive.exists():
        success = download_file(url, archive)
        if not success:
            return
    else:
        print(f"Archive already exists: {archive}")

    extract_archive(archive, folder / "extracted")


def download_dfdc_from_kaggle():
    """Download DFDC using an authorized Kaggle account."""
    folder = ROOT / "DFDC"
    folder.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            ["kaggle", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(
            "\nDFDC: Install Kaggle API and configure your credentials.\n"
            "Run: pip install kaggle\n"
            "Then configure your Kaggle API token."
        )
        return

    print("\nStarting DFDC download from Kaggle...")
    print("The Kaggle account must have accepted the competition rules.")

    try:
        subprocess.run(
            [
                "kaggle", "competitions", "download",
                "-c", "deepfake-detection-challenge",
                "-p", str(folder),
            ],
            check=True,
        )
        print("DFDC download completed. Extract archives if required.")
    except subprocess.CalledProcessError as error:
        print(
            "Kaggle download failed. Check access permissions, "
            f"credentials and competition rules. Details: {error}"
        )


def create_inventory():
    """Record downloaded files for reproducibility."""
    inventory = ROOT / "download_inventory.csv"

    with open(inventory, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["dataset", "relative_path", "size_bytes"])

        for path in ROOT.rglob("*"):
            if path.is_file() and path != inventory:
                writer.writerow([
                    path.parent.relative_to(ROOT).as_posix(),
                    path.relative_to(ROOT).as_posix(),
                    path.stat().st_size,
                ])

    print(f"\nInventory saved to: {inventory}")


def main():
    print(f"Dataset storage directory: {ROOT}")

    # Download DFDC using Kaggle if desired.
    # Uncomment the next line after configuring Kaggle access.
    # download_dfdc_from_kaggle()

    # Download other datasets only after authorized URLs are available.
    for name, config in DATASETS.items():
        if name == "DFDC" and not config["url"].strip():
            continue
        download_from_config(name, config)

    create_inventory()
    print("\nFinished. Review skipped downloads and error messages.")


if __name__ == "__main__":
    main()
