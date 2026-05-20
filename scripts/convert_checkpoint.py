"""Convert an audiocraft_trans .th checkpoint to safetensors format.

Run this script from inside the audiocraft_trans conda environment:

    python scripts/convert_checkpoint.py \\
        /lustre/.../01684fbb/checkpoint_350.th \\
        muscriptor_weights.safetensors

The script extracts the EMA weights from the `best_state` key and saves them
in safetensors format, which can be loaded by muscriptor.TranscriptionModel.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="Path to the .th checkpoint file")
    parser.add_argument("output", type=Path, help="Path for the output .safetensors file")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading checkpoint from {args.checkpoint} …")
    import torch
    pkg = torch.load(args.checkpoint, map_location="cpu")

    # The EMA weights live under 'best_state'
    if "best_state" not in pkg:
        print("Error: checkpoint has no 'best_state' key.", file=sys.stderr)
        print(f"Available keys: {list(pkg.keys())}", file=sys.stderr)
        sys.exit(1)

    best_state = pkg["best_state"]

    # best_state is either a dict of {name: tensor} or a nested structure.
    # Flatten if it contains a 'model' sub-key (common in audiocraft checkpoints).
    if isinstance(best_state, dict) and "model" in best_state:
        state_dict = best_state["model"]
    elif isinstance(best_state, dict):
        state_dict = best_state
    else:
        print("Error: unexpected structure of 'best_state'.", file=sys.stderr)
        sys.exit(1)

    # Ensure all tensors are float32 contiguous CPU tensors
    clean = {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            print(f"Skipping non-tensor key: {k}")
            continue
        clean[k] = v.float().contiguous().cpu()

    print(f"Extracted {len(clean)} tensors from best_state.")
    print("Saving to safetensors …")

    from safetensors.torch import save_file
    save_file(clean, str(args.output))
    print(f"Done. Saved to {args.output}")

    # Quick sanity check
    from safetensors.torch import load_file
    loaded = load_file(str(args.output))
    assert len(loaded) == len(clean), "Tensor count mismatch after round-trip!"
    print("Sanity check passed.")


if __name__ == "__main__":
    main()
