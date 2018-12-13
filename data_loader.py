# Preprocesses and cleans base kaggle data
# saves data to a csv in specified directory'''

import pandas as pd

# Specify directory to load and save data to/from
directory = '/Users/liamroberts/Desktop/Datasets/Rossmann/'

# Import Data
store = pd.read_csv(f'{directory}store.csv',
                    low_memory=False)

#state = pd.read_csv(f'{directory}store_states.csv',
                    #low_memory=False)

df = pd.read_csv(f'{directory}train.csv',
                 low_memory=False,
                 parse_dates=True,
                 index_col='Date')

# Encode Data Times
df['Month'] = df.index.month
df['WeekOfYear'] = df.index.weekofyear

# Merge Data Frames
df = pd.merge(df,
              store,
              on = 'Store')

#df = pd.merge(df,
              #state,
              #on = 'Store')

# Remove days where sales are zero and when store is Closed
df = df[(df['Sales'] > 0)]
df = df[(df['Open'] == 1)]

# -----Feature Engineering------ #

# Store Specific Metrics
agg_df = df.groupby(by='Store').mean()
agg_df['SalesPerCustomer'] = agg_df['Sales']/agg_df['Customers']
agg_df['MeanStoreSales'] = agg_df['Sales']
agg_df['MeanStoreCustomers'] = agg_df['Customers']
agg_df = agg_df[['MeanStoreSales',
                 'MeanStoreCustomers',
                 'SalesPerCustomer']]
df = pd.merge(df,
              agg_df,
              on='Store')

ratios = df.groupby('Store').mean()

# Promo Sales Ratio by Store
promo = df[df['Promo'] == 1].groupby('Store').mean()
no_promo = df[df['Promo'] == 0].groupby('Store').mean()
ratios['PromoRatio'] = promo.Sales/no_promo.Sales

# Promo Days Ratio by Store
promo = df[df['Promo'] == 1].groupby('Store').sum()
no_promo = df[df['Promo'] == 0].groupby('Store').sum()
ratios['PromoDaysRatio'] = promo.Open/no_promo.Open

# Holiday Ratio by Store
holiday = df[(df.SchoolHoliday == 1) | (df.StateHoliday == 1)].groupby(by='Store').mean()
no_holiday = df[(df.SchoolHoliday == 0) & (df.StateHoliday == 0)].groupby(by='Store').mean()
ratios['HolidayRatio'] = holiday.Sales/no_holiday.Sales

ratio_cols = ['PromoRatio',
              'PromoDaysRatio',
              'HolidayRatio']

ratios = ratios[ratio_cols]

df = pd.merge(df,
              ratios,
              on='Store')

df.drop(columns=['Promo2SinceWeek',
                 'Promo2SinceYear'],
        inplace=True)

# ------Save Data to Csv------ #

# Train test split ~90% through data
split_point = int(len(df)*0.9)
train = df[:split_point]
test = df[split_point:]

# Specify save folder
save_path = '/Users/liamroberts/Desktop/Datasets/Rossmann/'

# Save files
train.to_csv(f'{save_path}train_data.csv',
             index=False)

test.to_csv(f'{save_path}test_data.csv',
            index=False)

print("~Files Successfully Saved~")
