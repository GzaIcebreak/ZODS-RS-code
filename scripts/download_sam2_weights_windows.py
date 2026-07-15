import os
import urllib.request
from pathlib import Path


def download_file(url: str, output_path: str):
    """Download file with progress bar (Windows-friendly)"""
    print(f"Downloading {url} ...")
    print(f"Saving to: {output_path}")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            print(f"\rProgress: {percent:.1f}% ({downloaded // (1024*1024)} MB / {total_size // (1024*1024)} MB)", end="")
        else:
            print(f"\rProgress: {downloaded // (1024*1024)} MB", end="")
    
    urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
    print("\nDownload completed!")


def main():
    # SAM-2.0 weights (compatible with current repo sam2/ code)
    weights = {
        "sam2_hiera_tiny.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
        "sam2_hiera_small.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt", 
        "sam2_hiera_base_plus.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
        "sam2_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
    }
    
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)
    
    print("Available SAM-2.0 weights (compatible with this repo):")
    for i, (name, url) in enumerate(weights.items(), 1):
        print(f"{i}. {name}")
    
    choice = input("\nEnter number to download (or 'all' for all weights): ").strip()
    
    if choice.lower() == "all":
        to_download = list(weights.items())
    else:
        try:
            idx = int(choice) - 1
            name, url = list(weights.items())[idx]
            to_download = [(name, url)]
        except (ValueError, IndexError):
            print("Invalid choice")
            return
    
    for name, url in to_download:
        output_path = checkpoints_dir / name
        if output_path.exists():
            print(f"File {output_path} already exists, skipping.")
            continue
        download_file(url, str(output_path))
    
    print(f"\nAll downloads completed. Files saved to: {checkpoints_dir.absolute()}")


if __name__ == "__main__":
    main()
