"""Shared CLI plumbing: make the repo importable and parse common flags."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def base_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--task", help="task name -> configs/<task>.yaml")
    g.add_argument("--config", help="path to a YAML config")
    ap.add_argument("--quick", action="store_true", help="small data subset, < 5 min on CPU")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    return ap


def resolve_config(args):
    from core.config import config_for_task, load_config

    path = args.config or config_for_task(args.task or "classification")
    return load_config(path, quick=args.quick)
