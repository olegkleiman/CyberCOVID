import numpy as np

class EnumeratedDates:
    def __init__(self, dates_labels):
        self.labels = dates_labels

        enumerated_dates = np.array([])  # start with empty array
        for idx, x in np.ndenumerate(dates_labels):
            enumerated_dates = np.append(enumerated_dates, idx)

        self.indices = enumerated_dates

    def __enter__(self):
        return self.indices

    def __exit__(self, exc_type, exc_value, tb):
        return True
