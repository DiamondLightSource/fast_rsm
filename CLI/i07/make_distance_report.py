import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import papermill as pm  # pip install papermill
from fast_rsm.distance_calculator import parse_yaml

# yaml_file = (
#     "/dls/science/users/rpy65944/I07_work/dev_fast_rsm/testing_scripts/dist_522165.yaml"
# )


def convert_nb(nbpath: str, htmlpath: str, cwdpath: Path):
    # ---------- convert executed notebook to HTML ----------
    subprocess.run(  # [web:23]
        [
            "jupyter",
            "nbconvert",
            "--to",
            "html",
            "--output",
            htmlpath,
            nbpath,
        ],
        cwd=str(cwdpath),  # run in this directory
        check=True,
    )


def run_dist_calc(yaml_file: str):
    # Reading data from the YAML fil
    params = parse_yaml(yaml_file)
    now = datetime.now()

    template_nb = "/dls/science/users/rpy65944/I07_work/dev_fast_rsm/fast_rsm/CLI/i07/distance_report.ipynb"
    out_stem = Path(params["nexusfile"]).stem
    executed_nb = (
        Path(params["outdir"]) / f"{out_stem}_{now.strftime('%Y%m%d_%H%M')}.ipynb"
    )
    executed_html = (
        Path(params["outdir"]) / f"{out_stem}_{now.strftime('%Y%m%d_%H%M')}.html"
    )
    # ---------- run notebook with papermill ----------
    pm.execute_notebook(  # [web:26][web:31]
        input_path=str(template_nb),
        output_path=str(executed_nb),
        parameters=params,
        kernel_name="python3",
    )

    project_root = Path(__file__).resolve().parent.parent

    convert_nb(str(executed_nb), str(executed_html), project_root)
    subprocess.run(["firefox", str(executed_html)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-dist",
        "--dist_yaml",
        help="path to yaml file with distance calculation settings",
    )

    args = parser.parse_args()
    run_dist_calc(args.dist_yaml)
