import numpy as np
import pandas as pd
import datetime

class EnumeratedDates:
    def __init__(self, dates_labels):
        self.labels = dates_labels

        enumerated_dates = np.array([])  # start with empty array
        for idx, x in np.ndenumerate(dates_labels):
            enumerated_dates = np.append(enumerated_dates, idx)

        self.indices = enumerated_dates
        # self.objects = [pd.to_datetime(d, format='%d/%m/%Y').to_pydatetime() for d in self.labels]
        self.objects = [datetime.datetime.strptime(d, '%d/%m/%Y') for d in self.labels]

    def __enter__(self):
        return self.indices

    def __exit__(self, exc_type, exc_value, tb):
        return True
