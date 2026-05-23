"""AEMO NEM data loader — STUB.

The data team owns this module. It needs to provide three CLI commands and
one library function, all called from docs/creating-scenarios.md:

    python -m watt_the_hack.data_loaders.aemo download   --region SA --date 2024-04-12 --out watt_the_hack/data/aemo/raw/
    python -m watt_the_hack.data_loaders.aemo preprocess --region SA --date 2024-04-12 --scaling scale_up_engine
    python -m watt_the_hack.data_loaders.aemo inspect    watt_the_hack/data/aemo/SA_2024-04-12.parquet

    load_aemo_parquet(region: str, date: str) -> dict[str, list[float]]

What each does:

  download   — pull raw 5-min CSVs from NEMWEB into watt_the_hack/data/aemo/raw/
                Sources: DispatchIS_Reports/, Dispatch_SCADA/, ROOFTOP_PV/ACTUAL/
  preprocess — resample 5-min → 15-min, sum rooftop PV + grid-scale solar,
                apply scaling mode (scale_up_engine | scale_down_data),
                write watt_the_hack/data/aemo/{region}_{date}.parquet
                with columns: timestamp, demand, solar, price
  inspect    — print row count, column ranges, ASCII plot for sanity-check
  load_aemo_parquet — read the preprocessed file, return profile dict

Until this exists, scenarios with data_source='aemo' fail loudly via
watt_the_hack.data_loaders.scenarios._load_aemo.
"""

from __future__ import annotations


def load_aemo_parquet(region: str, date: str) -> dict[str, list[float]]:
    raise NotImplementedError(
        "AEMO loader not implemented. See module docstring for the spec."
    )


if __name__ == "__main__":
    import sys

    print(
        "AEMO loader is a stub. See watt_the_hack/data_loaders/aemo.py docstring for "
        "the commands the data team needs to build.",
        file=sys.stderr,
    )
    sys.exit(1)
