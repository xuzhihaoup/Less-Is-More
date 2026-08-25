"""Command-line entry point for model training."""

from .config import parse_config
from .engine import train


def main() -> None:
    train(parse_config())


if __name__ == "__main__":
    main()
