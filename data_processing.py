from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
import pandas as pd

def handle_imbalanced_data(X, y):
    rdm = RandomOverSampler()
    X_resampled, y_resampled = rdm.fit_resample(X, y)
    dta_resampled = pd.concat([X_resampled, y_resampled], axis=1)
    return dta_resampled


def scale_data(X):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

