import numpy as np
import pmdarima as pm
from pyts.approximation import PiecewiseAggregateApproximation
from pyts.image import RecurrencePlot
from rcnn.utils import load_from_tsfile_to_dataframe
import time


class DataLoader:
    def __init__(self, path_to_file):
        self.data, _ = load_from_tsfile_to_dataframe(path_to_file=path_to_file)
        self.num_instances, self.num_variables = self.data.shape

        # Use first 20k samples only to save memory
        if self.num_instances > 20000:
            self.data = self.data.iloc[:20000]
            self.num_instances = 20000

        # Get (max) time series length.
        # Handle case where variables do not have the same lengths by taking the max
        # Shorter ones are padded
        self.ts_length = max([len(self.data.iloc[0][i]) for i in range(self.num_variables)])

        # If no. of observations > 2000, use the first 2000 observations only
        self.ts_length = self.ts_length if self.ts_length <= 2000 else 2000

    def load_data(self):
        """
        Prepare numpy array X with shape (num_instances, ts_length, num_variables)
        and Y with shape (num_instances, num_variables)
        """

        # Decrement by 1 since observation s_j, 1 <= j <= t is split as such:
        # X_i = [s_1,...,s_t-1], Y_i = s_t
        X, Y = np.empty((self.num_instances, self.ts_length - 1, self.num_variables)), \
               np.empty((self.num_instances, self.num_variables))

        # For all instance
        start = time.time()
        for idx, row in enumerate(self.data.iterrows()):
            for i in range(self.num_variables):
                # Get current variable's series
                # Apply linear interpolation on missing values
                # Handle case when no. observations > 2000 by enforcing a length slice
                s = row[1][i].interpolate(limit_direction='both').to_numpy()[:self.ts_length]

                # Case when a variable's series has a shorter length
                if s.size != self.ts_length:
                    # Pad beginning with zeros
                    s = np.pad(s, (self.ts_length - s.size, 0), 'constant', constant_values=0.)

                X[idx, :, i] = s[:-1]
                Y[idx, i] = s[-1]
        end = time.time()
        # print(f"Data loaded in {end - start} seconds")

        # Free data variable
        self.data = None

        return X, Y

    def get_residuals(self):
        """
        Get ARIMA residuals of each variable. Used for BDS tests
        """
        # Get time series length.
        # Handle case where variables do not have the same lengths by taking the max
        # Shorter ones are padded
        self.ts_length = max([len(self.data.iloc[0][i]) for i in range(self.num_variables)])

        residuals = np.empty((self.num_variables, self.ts_length))

        # Take a sample. For each variable
        for i in range(self.num_variables):
            # Obtain variable's time series
            sample = self.data.iloc[0][i].interpolate(limit_direction='both').to_numpy()

            # Fit arima and obtain residuals
            model = pm.auto_arima(sample, seasonal=False, start_p=2, max_p=10, max_d=10, max_q=10)
            res = np.array(model.resid())

            # Case when a variable's series has a shorter length
            if res.size != self.ts_length:
                # Pad beginning with zeros
                res = np.pad(res, (self.ts_length - res.size, 0), 'constant', constant_values=0.)

            residuals[i] = res

        return residuals


class CNNDataLoader(DataLoader):
    def __init__(self, path_to_file, img_size):
        super().__init__(path_to_file)
        self.img_size = img_size

    def load_data(self):
        """
        Prepare numpy array X with shape (num_instances, img_size, img_size, num_variables)
        and y with shape (num_instances, num_variables)
        """

        X, Y = np.empty((self.num_instances, self.img_size, self.img_size, self.num_variables)), \
               np.empty((self.num_instances, self.num_variables))
        print(X.shape)

        # Initialize PAA transformer
        paa = PiecewiseAggregateApproximation(window_size=None, output_size=self.img_size, overlapping=False)
        rp = RecurrencePlot()

        # For all instance
        start = time.time()
        for idx, row in enumerate(self.data.iterrows()):
            for i in range(self.num_variables):
                # Get current variable's series
                # Apply linear interpolation on missing values
                s = row[1][i].interpolate(limit_direction='both').to_numpy()[:self.ts_length]
                # Apply PAA and RP
                X[idx, :, :, i] = rp.transform(paa.transform(np.expand_dims(s[:-1], axis=0)))[0]
                Y[idx, i] = s[-1]
        end = time.time()
        print(f"Data loaded in {end - start} seconds")

        return X, Y
