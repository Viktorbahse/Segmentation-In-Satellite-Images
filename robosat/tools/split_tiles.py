#!/usr/bin/env python3
"""
split_tiles.py

Usage:
  python split_tiles.py --input tiles.csv [--train 0.8 --val 0.1 --holdout 0.1] [--seed 42] [--shuffle]

Default ratios: 0.8 0.1 0.1
Outputs: train.csv, val.csv, holdout.csv (in same folder as input)
"""
import argparse
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

def parse_args():
    p = argparse.ArgumentParser(description="Split a CSV into train/val/holdout.")
    p.add_argument("--input", "-i", required=True, help="Input CSV file (tiles.csv).")
    p.add_argument("--train", type=float, default=0.8, help="Train ratio (default 0.8).")
    p.add_argument("--val", type=float, default=0.1, help="Validation ratio (default 0.1).")
    p.add_argument("--holdout", type=float, default=0.1, help="Hold-out ratio (default 0.1).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    p.add_argument("--shuffle", action="store_true", help="Shuffle rows before splitting (default False).")
    return p.parse_args()

def main():
    args = parse_args()

    ratios = (args.train, args.val, args.holdout)
    if any(r < 0 for r in ratios):
        print("Error: ratios must be non-negative.", file=sys.stderr); sys.exit(1)
    s = sum(ratios)
    if s <= 0:
        print("Error: sum of ratios must be > 0.", file=sys.stderr); sys.exit(1)
    # Normalize ratios to sum 1
    train_ratio, val_ratio, holdout_ratio = (r / s for r in ratios)

    df = pd.read_csv(args.input)
    if args.shuffle:
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # First split: train vs temp (val+holdout)
    if train_ratio <= 0:
        train_df = df.iloc[0:0].copy()
        temp_df = df
    elif train_ratio >= 1.0:
        train_df = df
        temp_df = df.iloc[0:0].copy()
    else:
        train_df, temp_df = train_test_split(df, train_size=train_ratio, random_state=args.seed, shuffle=not args.shuffle)

    # If there is no remaining data for val/holdout, make them empty
    if val_ratio + holdout_ratio <= 0 or temp_df.shape[0] == 0:
        val_df = temp_df.iloc[0:0].copy()
        holdout_df = temp_df.iloc[0:0].copy()
    else:
        # Proportion of val within temp
        val_within_temp = val_ratio / (val_ratio + holdout_ratio)
        if val_within_temp <= 0:
            val_df = temp_df.iloc[0:0].copy()
            holdout_df = temp_df
        elif val_within_temp >= 1:
            val_df = temp_df
            holdout_df = temp_df.iloc[0:0].copy()
        else:
            val_df, holdout_df = train_test_split(temp_df, train_size=val_within_temp, random_state=args.seed+1, shuffle=not args.shuffle)

    # Output filenames
    base_dir = os.path.dirname(os.path.abspath(args.input))
    train_out = os.path.join(base_dir, "train.csv")
    val_out = os.path.join(base_dir, "val.csv")
    holdout_out = os.path.join(base_dir, "test.csv")

    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    holdout_df.to_csv(holdout_out, index=False)

    print(f"Wrote {train_out} ({len(train_df)} rows), {val_out} ({len(val_df)} rows), {holdout_out} ({len(holdout_df)} rows).")

if __name__ == "__main__":
    main()
