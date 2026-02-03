import argparse
import csv
import json

from supermercado import burntiles
from tqdm import tqdm

import rootutils
rootutils.setup_root(__file__, indicator="robosat", pythonpath=True)


def add_parser(subparser):
    parser = subparser.add_parser(
        "cover",
        help="generates tiles covering GeoJSON features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--zoom", type=int, required=True, help="zoom level of tiles")
    parser.add_argument("features", type=str, help="path to GeoJSON features")
    parser.add_argument("out", type=str, help="path to csv file to store tiles in")

    parser.set_defaults(func=main)


def main(args):
    with open(args.features) as f:
        features = json.load(f)

    tiles = []

    for feature in tqdm(features["features"], ascii=True, unit="feature"):
        tiles.extend(map(tuple, burntiles.burn([feature], args.zoom).tolist()))

    # tiles can overlap for multiple features; unique tile ids
    tiles = list(set(tiles))

    with open(args.out, "w") as fp:
        writer = csv.writer(fp)
        writer.writerows(tiles)


if __name__ == "__main__":
    class Args:
        def __init__(self, feature):
            self.zoom = 20
            self.features = feature
            self.out = "data/tiles.csv"
    
    import glob, os
    candidates = glob.glob("data/*buildings*.geojson")
    if not candidates:
        candidates = glob.glob("data/*.geojson")
    chosen = candidates[0]  
    args = Args(chosen)
    main(args)