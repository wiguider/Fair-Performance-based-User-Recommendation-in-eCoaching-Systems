import pandas as pd


class Dataset(object):
    """
    reads a dataset from a csv-file into a dataframe
    """

    @property
    def data(self):
        """
        a data frame that contains the dataset to be analyzed
        """
        return self.__data

    def __init__(self, data):
        """
        Constructor
        """
        if isinstance(data, str):
            # expect data to be a filename, engine=python enables auto-detection of separator
            self.__data = pd.read_csv(data, sep=',', quotechar='"', low_memory=False, memory_map=True)
        elif isinstance(data, pd.DataFrame):
            self.__data = data
        self.normalized_columns = {'header': ['mean', 'min', 'max']}

    def normalize_column(self, column_name):
        mean_col = self.data[column_name].dropna().mean()
        min_col = self.data[column_name].dropna().min()
        max_col = self.data[column_name].dropna().max()
        self.normalized_columns[column_name] = [mean_col, min_col, max_col]
        self.data[column_name] = self.data[column_name].apply(lambda x: (x - mean_col) / (max_col - min_col))

    def normalize_columns(self, columns):
        for column_name in columns:
            self.normalize_column(column_name)
        print(self.compute_statistic(columns))

    def remove_columns(self, columns):
        self.__data = self.data.drop(columns, axis=1)

    def remove_columns_with_single_value(self):
        nunique = self.data.apply(pd.Series.nunique)
        cols_to_drop = nunique[nunique == 1].index
        self.__data = self.data.drop(cols_to_drop, axis=1)
        print(cols_to_drop)

    def dropna(self,axis=0):
        self.__data = self.data.dropna(axis=axis)

    def compute_statistic(self, selected_atts):
        """
        Compute the statistics of input attributes.

        Attributes:
            selected_atts: array that stores the attributes to be computed
        Return:  json data of computed statistics
        """
        output_df = pd.DataFrame(columns=["attribute", "median", "mean", "min", "max"])
        for atti in selected_atts:
            atti_stats = self.__data.describe().loc[["50%", "mean", "min", "max"], atti].tolist()
            output_df.loc[output_df.shape[0]] = [atti] + atti_stats
        return output_df
    
    def set_column(self, column_name, new_column_data):
        self.__data[column_name] = new_column_data
