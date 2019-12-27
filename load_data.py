import pandas as pd


df_csv = pd.read_csv('data.csv', header=0)
data = df_csv[['dl_RealHP', 'dl_RealDI', 'dl_TotLoanVal_sa', 'dl_HouseStock', 'dl_ltv']].iloc[1:, :]
