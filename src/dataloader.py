from batch import (
    BatchAE,
    BatchMasking,
    BatchSubstructContext,
    BatchSubstructContext3D,
    BatchHybrid,
)
from torch_geometric.data import Batch
from torch.utils.data import DataLoader


class DataLoaderSubstructContext(DataLoader):
    """Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)"""

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderSubstructContext, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchSubstructContext.from_data_list(
                data_list
            ),
            **kwargs
        )


class DataLoaderSubstructContext3D(DataLoader):
    """Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)"""

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderSubstructContext3D, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchSubstructContext3D.from_data_list(
                data_list
            ),
            **kwargs
        )


class DataLoaderMasking(DataLoader):
    """Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)"""

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderMasking, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchMasking.from_data_list(data_list),
            **kwargs
        )


class DataLoaderAE(DataLoader):
    """Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)"""

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderAE, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchAE.from_data_list(data_list),
            **kwargs
        )


class DataLoaderHybrid(DataLoader):
    """Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)"""

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderHybrid, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: (
                [
                    BatchHybrid.from_data_list([_[0] for _ in data_list]),
                    Batch.from_data_list([_[1] for _ in data_list]),
                    Batch.from_data_list([_[2] for _ in data_list]),
                ]
                if isinstance(data_list[0], tuple) or isinstance(data_list[0], list)
                else BatchHybrid.from_data_list(data_list)
            ),
            **kwargs
        )
