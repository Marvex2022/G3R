import pytest

from train import _map_rank_to_cuda_device


def test_two_processes_are_mapped_to_each_of_two_gpus():
    mapped = [
        _map_rank_to_cuda_device(rank, 4, 2, 2)
        for rank in range(4)
    ]
    assert mapped == [0, 0, 1, 1]


def test_mapping_rejects_inconsistent_world_size():
    with pytest.raises(ValueError, match="WORLD_SIZE=3"):
        _map_rank_to_cuda_device(0, 3, 2, 2)
