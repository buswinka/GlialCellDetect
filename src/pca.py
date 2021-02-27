import torch


def pca(a: torch.Tensor, n_dim: int = 2) -> torch.Tensor:
    """
    From pytorch example. Small helper function to perform a PCA and project on
    the N_dim relevant dimmensions for clustering

    :param a:
    :param n_dim:
    :return:
    """
    assert a.ndim == 2
    (U, S, V) = torch.pca_lowrank(a)
    return torch.matmul(a, V[:, :n_dim])
