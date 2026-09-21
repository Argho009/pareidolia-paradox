import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import colorama
from colorama import Fore, Style

from pareidolia.utils import check_dependencies, Config, setup_logger, set_seed


def main():
    colorama.init(autoreset=True)
    print(f"""
{Fore.CYAN}╔══════════════════════════════════════════╗
{Fore.YELLOW}║    THE PAREIDOLIA PARADOX  —  Pipeline   ║
{Fore.CYAN}╚══════════════════════════════════════════╝{Style.RESET_ALL}
""")
    check_dependencies()

    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--gui",     action="store_true")
    g.add_argument("--train",   action="store_true")
    g.add_argument("--predict", action="store_true")
    g.add_argument("--image",   type=str, help="Predict on a single image")
    args = p.parse_args()

    cfg = Config.load("config.json")
    os.makedirs(cfg.output_dir, exist_ok=True)
    setup_logger(cfg.output_dir)
    set_seed(cfg.seed)

    if args.train:
        from pareidolia.train import train_pipeline
        train_pipeline(cfg)
    elif args.predict:
        from pareidolia.predict import predict_pipeline
        predict_pipeline(cfg)
    elif args.image:
        from pareidolia.predict import predict_single_image
        predict_single_image(cfg, args.image)
    else:
        from pareidolia.ui.app import launch_gui
        launch_gui()


if __name__ == "__main__":
    main()
