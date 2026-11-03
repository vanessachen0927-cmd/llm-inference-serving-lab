"""CPU-only validation tests; do not import or download torch."""
import pytest
from scripts.distributed_allreduce import (
    choose_backend, expected_sum, validate_configuration,
)


@pytest.mark.parametrize("world_size, expected", [(1, 1.0), (2, 3.0), (4, 10.0)])
def test_expected_allreduce_result(world_size, expected):
    assert expected_sum(world_size) == expected


def test_invalid_world_size():
    with pytest.raises(ValueError):
        expected_sum(0)


def test_backend_selection():
    assert choose_backend("auto", False) == "gloo"
    assert choose_backend("auto", True) == "nccl"
    assert choose_backend("gloo", True) == "gloo"
    assert choose_backend("nccl", True) == "nccl"
    with pytest.raises(ValueError):
        choose_backend("mpi", False)


def test_single_gpu_is_valid_for_one_rank_nccl_smoke():
    validate_configuration("nccl", 1, 0, 1, True)


def test_second_nccl_rank_cannot_share_single_gpu():
    with pytest.raises(RuntimeError, match="distinct visible CUDA GPU"):
        validate_configuration("nccl", 2, 1, 1, True)


def test_nccl_requires_cuda():
    with pytest.raises(RuntimeError):
        validate_configuration("nccl", 1, 0, 0, False)


def test_cpu_gloo_does_not_require_gpu():
    validate_configuration("gloo", 2, 1, 0, False)


def test_invalid_rank_rejected():
    with pytest.raises(ValueError):
        validate_configuration("gloo", 2, -1, 0, False)
