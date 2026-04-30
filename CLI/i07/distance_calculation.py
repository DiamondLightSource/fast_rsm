import argparse

from fast_rsm.distance_calculator import calc_distance, parse_yaml


def parse_and_run(yaml_file: str):
    params = parse_yaml(yaml_file)

    calc_distance(**params)


if __name__ == "__main__":
    HELP_STR = (
        "takes in paths to file and central pixel perpendicular to detector movement direction,\n"
        + "finds signals in images and calculates distance based on pixel shifts and angles"
    )
    parser = argparse.ArgumentParser(description=HELP_STR)

    parser.add_argument(
        "-dist",
        "--dist_yaml",
        help="path to yaml file with distance calculation settings",
    )

    args = parser.parse_args()
    parse_and_run(args.dist_yaml)
