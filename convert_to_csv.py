#!/usr/bin/env python3

import argparse
import os

import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description="Example parser script."
    )

    # Positional argument
    parser.add_argument(
        "input",
        help="Path to the input JSON file"
    )

    # Optional argument with value
    parser.add_argument(
        "-o", "--output",
        help="Path to output CSV file",
        default="output.txt"
    )

    parser.add_argument(
        "--idx",
        type=int,
        help="Architecture index"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed"
    )
        
    parser.add_argument(
        "--dataset",
        help="Dataset",
        default="cifar10"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("Arguments:", args)

    df = pd.read_json(args.input, orient='index').T

    df["seed"] = args.seed
    df["idx"] = args.idx
    df["dataset"] = args.dataset

    print(df)

    if os.path.exists(args.output):
        mode = "a"
        header = False
    else:
        mode = "w"
        header = True

    df.to_csv(args.output, mode = mode, header = header, index = False)


if __name__ == "__main__":
    main()