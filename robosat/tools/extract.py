import argparse

import rootutils
rootutils.setup_root(__file__, indicator="robosat", pythonpath=True)

from robosat.osm.parking import ParkingHandler # noqa
from robosat.osm.building import BuildingHandler # noqa
from robosat.osm.road import RoadHandler # noqa

# Register your osmium handlers here; in addition to the osmium handler interface
# they need to support a `save(path)` function for GeoJSON serialization to a file.
handlers = {"parking": ParkingHandler, "building": BuildingHandler, "road": RoadHandler}


def add_parser(subparser):
    parser = subparser.add_parser(
        "extract",
        help="extracts GeoJSON features from OpenStreetMap",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--type", type=str, required=True, choices=handlers.keys(), help="type of feature to extract")
    parser.add_argument("--batch", type=int, default=100000, help="number of features to save per file")
    parser.add_argument("map", type=str, help="path to .osm.pbf base map")
    parser.add_argument("out", type=str, help="path to GeoJSON file to store features in")

    parser.set_defaults(func=main)
    return parser

def main(args):
    handler = handlers[args.type](args.out, args.batch)
    handler.apply_file(filename=args.map, locations=True)
    handler.flush()

if __name__ == "__main__":
    handler = BuildingHandler("data/buildings.geojson", 100_000_000)
    handler.apply_file(filename="data/map.osm.pbf", locations=True)
    handler.flush()