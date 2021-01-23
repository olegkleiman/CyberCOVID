import numpy as np

class EnumeratedDates:
    def __init__(self, dates_labels):
        self.labels = dates_labels

        enumeratedDates = np.array([])  # start with empty array
        for idx, x in np.ndenumerate(dates_labels):
            enumeratedDates = np.append(enumeratedDates, idx)

        self.indices = enumeratedDates

    def __enter__(self):
        return self.indices

    def __exit__(self, exc_type, exc_value, tb):
        return True
