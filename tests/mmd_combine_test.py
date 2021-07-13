from mmd_wrapper import mmd_combine


def test_mmd_combine_running():
    assert mmd_combine(method='_running')
