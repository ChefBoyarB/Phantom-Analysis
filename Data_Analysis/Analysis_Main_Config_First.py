"""
Use this entry point for normal analysis runs.
It requires a JSON config file and delegates to Analysis_Main_Engine.py.
"""

import argparse

from Analysis_Main_Engine import load_settings_from_config, main


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the main transport analysis using a required JSON config file."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a JSON config file under configs/analysis or another valid location.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_info = load_settings_from_config(args.config)
    print(f"Loaded config file: {config_info['config_path']}")
    if config_info.get("config_name"):
        print(f"Config name: {config_info['config_name']}")
    main(config_info=config_info)
