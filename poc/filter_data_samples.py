"""
This script processes a directory containing .tar archives, each of which may contain JSON
metadata and associated files. It identifies and extracts files whose metadata entries
match a specific condition.

Assumptions:
    - The input directory contains one or more .tar files.
    - Each .tar file, when extracted, contains a JSON file with a top-level "Files" list, where each element (file_data) includes:
        {
            "Filename": "<filename>",
            "Prompt": {
                "Category Description": "<category name>",
                ...
            },
            ...
        }
    - Files referenced by "Filename" are located in the same directory as their JSON file.

Outputs:
    - Moved files in: ./digital_assistant_prompt_samples/
    - Metadata file:  ./digital_assistant_prompt_samples/digital_assistant_metadata.json
"""

import os
import tarfile
import json
import shutil
from pathlib import Path

def extract_ids_from_tar(input_dir: str,):
    """
    Extract IDs and file_data from all JSON files inside .tar archives in a directory
    where the JSON contains {key: value}.
    If a .tar has already been extracted, skip re-extraction.
    Move matching files to 'digital_assistant_prompts' folder.
    """

    input_dir = Path(input_dir)
    all_metadata = []

    # Folder to store moved prompt files
    da_prompt_dir = input_dir / "digital_assistant_prompt_samples"
    da_prompt_dir.mkdir(exist_ok=True)

    for tar_path in input_dir.glob("*.tar"):
        print(f"\nProcessing {tar_path.name}...")
        # Temporary folder for extraction
        temp_dir = tar_path.parent / f"{tar_path.stem}_extracted"

        # Check if already extracted
        if temp_dir.exists() and any(temp_dir.iterdir()):
            print(f"Already extracted: {temp_dir}")
        else:
            print(f"Extracting {tar_path.name}...")
            temp_dir.mkdir(exist_ok=True)
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(temp_dir)

        # Search for JSON files inside extracted folder
        for json_file in temp_dir.rglob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for file_data in data.get("Files", []):
                    prompt = file_data.get("Prompt", {})
                    if prompt.get("Category Description") == "Digital Assistant Commands":
                        filename = file_data.get("Filename")
                        if filename:
                            all_metadata.append(file_data)

                            # Move matching file to new folder
                            src_path = json_file.parent / filename
                            if src_path.exists():
                                dst_path = da_prompt_dir / filename
                                shutil.move(str(src_path), str(dst_path))
                            else:
                                print(f"File not found for moving: {src_path}")
            except Exception as e:
                print(f"Error reading {json_file}: {e}")

    # Write all matching file_data entries to metadata JSON
    metadata_file = da_prompt_dir / "digital_assistant_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as meta_out:
        json.dump(all_metadata, meta_out, indent=2)

    print(f"Moved matching files into: {da_prompt_dir}")
    print(f"Metadata written to: {metadata_file}")


if __name__ == "__main__":
    input_dir = './'
    extract_ids_from_tar(input_dir)
