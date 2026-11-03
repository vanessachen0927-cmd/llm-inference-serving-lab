"""CPU-only argument validation tests; no model weights are downloaded."""
from argparse import Namespace
import pytest
from scripts.diffusers_inference import DEFAULT_MODEL, build_parser, validate_args


def test_default_configuration():
    args = build_parser().parse_args([])
    validate_args(args)
    assert args.model == DEFAULT_MODEL
    assert args.steps == 15
    assert args.width == args.height == 384


@pytest.mark.parametrize("kwargs", [
    {"steps": 0}, {"steps": 101}, {"width": 250}, {"height": 769},
    {"seed": -1}, {"model": " "}, {"prompt": " "},
    {"output": "same.png", "metadata": "same.png"},
])
def test_invalid_settings(kwargs):
    args = vars(build_parser().parse_args([]))
    args.update(kwargs)
    with pytest.raises(ValueError):
        validate_args(Namespace(**args))


def test_override_settings():
    args = build_parser().parse_args([
        "--steps", "10", "--width", "256", "--height", "512", "--seed", "123"
    ])
    validate_args(args)
    assert args.steps == 10 and args.width == 256 and args.height == 512
