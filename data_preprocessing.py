import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices

def load_data():
    dta = sm.datasets.fair.load_pandas().data
    dta['affair'] = (dta.affairs > 0).astype(int)
    return dta

def preprocess_data(dta):
    y, X = dmatrices('affair ~ rate_marriage + age + yrs_married + children + religious + educ + C(occupation) + C(occupation_husb)', dta, return_type="dataframe")
    X = X.rename(columns={
        'C(occupation)[T.2.0]':'occ_2',
        'C(occupation)[T.3.0]':'occ_3',
        'C(occupation)[T.4.0]':'occ_4',
        'C(occupation)[T.5.0]':'occ_5',
        'C(occupation)[T.6.0]':'occ_6',
        'C(occupation_husb)[T.2.0]':'occ_husb_2',
        'C(occupation_husb)[T.3.0]':'occ_husb_3',
        'C(occupation_husb)[T.4.0]':'occ_husb_4',
        'C(occupation_husb)[T.5.0]':'occ_husb_5',
        'C(occupation_husb)[T.6.0]':'occ_husb_6'
    })
    dta = pd.concat([X, y], axis=1)
    return dta
