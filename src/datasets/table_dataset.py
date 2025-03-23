import pandas as pd
import torch
from torch.utils.data import Dataset


class TableDataset(Dataset):
    """
    Class for the table datasets
    """

    def __init__(
        self,
        path,
        discrete_columns,
        limit=None,
        shuffle=False,
        instance_transforms=None,
    ):
        """
        Args:
            path (str): path to csv containing table dataset
            discrete_columns (list[str]): the names of discrete columns in the dataset
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle (bool): if True, shuffle the table. Uses numpy random package.
            instance_transforms (dict[Callable] | None): transforms that
                should be applied on the instance. Depend on the
                tensor name.
        """
        table = pd.read_csv(path)
        table = self._shuffle_and_limit_table(table, limit, shuffle)
        self.table = table
        self.instance_transforms = instance_transforms
        self.discrete_columns = discrete_columns

    def __len__(self):
        """
        Get length of the dataset.
        """
        return self.table.shape[0]

    def __getitem__(self, ind):
        """
        Get element from the table, preprocess it, and combine it into a dict.

        Args:
            ind (int | np.array): index, of array of indices, in the self.table list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        data = torch.tensor(self.table.iloc[ind].to_numpy()).float()
        # we assume that the index is the last column
        if len(data.shape) == 1:
            label = int(data[-1])
        else:
            label = data[:, -1].type(torch.long)
        instance_data = {"data_object": data, "labels": label}
        instance_data = self.preprocess_data(instance_data)
        return instance_data

    def preprocess_data(self, instance_data):
        """
        Preprocess data with instance transforms.

        Each tensor in a dict undergoes its own transform defined by the key.

        Args:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element) (possibly transformed via
                instance transform).
        """
        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
                instance_data[transform_name] = self.instance_transforms[
                    transform_name
                ](instance_data[transform_name])
        return instance_data

    @staticmethod
    def _shuffle_and_limit_table(table, limit, shuffle):
        """
        Shuffle elements in table and limit the total number of elements.

        Args:
            table (pd.DataFrame): Pandas dataframe
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle (bool): if True, shuffle the table. Uses numpy random package.
        """
        if shuffle:
            table = table.sample(frac=1)

        if limit is not None:
            table = table[:limit]
        return table
