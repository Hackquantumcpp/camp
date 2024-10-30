import pandas as pd
from functools import reduce
from pathlib import Path

urban_stats = pd.read_csv('data/fundamentals/urban_stats.csv')

hist_results = pd.read_csv('data/fundamentals/cpvi.csv')
hist_results = hist_results.rename({'State':'state'}, axis=1)

white_evangel = pd.read_csv('data/fundamentals/white_evangel_pct.csv')

acs_age = pd.read_csv('data/fundamentals/ACS_data.csv')
acs_age = acs_age.T.set_index([0]).T
median_ages = acs_age[['Geographic Area Name', 'Estimate!!Total!!Total population!!SUMMARY INDICATORS!!Median age (years)']]
median_ages = median_ages.rename({'Geographic Area Name':'state', 'Estimate!!Total!!Total population!!SUMMARY INDICATORS!!Median age (years)':'median_age'}, axis=1)
cd_median_ages = pd.read_csv('data/fundamentals/cd_median_ages.csv')
median_ages = pd.concat([median_ages, cd_median_ages], axis=0)


regions = pd.read_csv('data/fundamentals/regions_538.csv')

corr_feature_dfs = [urban_stats, hist_results, white_evangel, median_ages, regions]
state_features = reduce(lambda left, right: pd.merge(left, right, on=['state'], how='inner'), corr_feature_dfs)
state_features = state_features.rename({'pct':'white_evangel'}, axis=1)
state_features['college_ed'] = state_features['ed_undergrad'] + state_features['ed_grad']

features = ['state', 'pw_density', 'white', 'hispanic', 'black', 'aapi', 'native', 'dem_20', 'rep_20', 
            'dem_16', 'rep_16', 'white_evangel', 'college_ed', 'median_age', 'region']

corr_features = state_features[features]
corr_features = corr_features.set_index(['state'])
corr_features['median_age'] = corr_features['median_age'].astype(float)

# Region data engineering
regions_dict = {'South':0, 'Southeast':1, 'Northeast':2, 'Southwest':3, 'Rust Belt':4, 'Plains':5, 'New England':6, 'Tex-ish':7,
               'Pacific':8, 'Mountain':9}
corr_features['region'] = corr_features['region'].map(lambda x: regions_dict[x])

corr_features_std = (corr_features - corr_features.mean()) / corr_features.std()
corr_matrix = corr_features_std.T.corr(method='pearson')

corr_matrix = corr_matrix.reset_index().drop(['state'], axis=1)

filepath = Path('data/fundamentals/state_correlation_matrix.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
corr_matrix.to_csv(filepath, index=False)