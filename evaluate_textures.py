import argparse
from cleanfid import fid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FID between two directories using CleanFID.")
    parser.add_argument("path1", type=str, help="Path to the first image directory (e.g., ground truth).")
    parser.add_argument("path2", type=str, help="Path to the second image directory (e.g., generated).")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (default: cuda:0)")

    args = parser.parse_args()

    score = fid.compute_fid(args.path1, args.path2, device=args.device)
    print(f"FID score: {score}")