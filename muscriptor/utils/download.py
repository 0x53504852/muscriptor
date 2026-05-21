"""Weight download utility with caching in ~/.cache/muscriptor/."""

import typing as tp
from pathlib import Path


_CACHE_DIR = Path.home() / ".cache" / "muscriptor"


def download_if_necessary(url: str, filename: tp.Optional[str] = None) -> Path:
    """Download url to cache dir if not already present.

    Args:
        url: Direct download URL for the weights file.
        filename: Override the filename; defaults to the last path component of url.

    Returns:
        Path to the cached file.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        hf_hub_download = None

    if filename is None:
        filename = url.split("/")[-1].split("?")[0]

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = _CACHE_DIR / filename

    if dest.exists():
        return dest

    # Try huggingface_hub if the URL looks like a HF URL
    if hf_hub_download is not None and "huggingface.co" in url:
        # Parse repo_id and filename from HF URL pattern:
        # https://huggingface.co/{repo_id}/resolve/{revision}/{path}
        parts = url.replace("https://huggingface.co/", "").split("/resolve/")
        if len(parts) == 2:
            repo_id = parts[0]
            rev_and_file = parts[1].split("/", 1)
            revision = rev_and_file[0]
            hf_filename = rev_and_file[1] if len(rev_and_file) > 1 else filename
            cached = hf_hub_download(
                repo_id=repo_id,
                filename=hf_filename,
                revision=revision,
                cache_dir=str(_CACHE_DIR),
            )
            return Path(cached)

    # Generic download fallback
    import urllib.request

    print(f"Downloading {filename} …")
    urllib.request.urlretrieve(url, dest)
    return dest
