This dataset contains two .csv files that can be used as a new benchmark data for the solving of real-world sales forecasting problem. All data are real and obtained experimentally in production environment in one of the biggest retail company in Bosnia and Herzegovina.
The available data in this dataset are in period from 2014/03/01 to 2021/03/01. Data are aggregated on monthly basis for 50 top items of one very popular brand in 4 different organizational units.

Dataset is divided into two files:

1_target_ts.csv (data about quantities)
- item -- item code
- org -- organizational unit code
- date -- date of sale of the item in format [YYYY]-[MM]-[DD]. First day in a month.
- quantity -- quantity of sale of the item in the whole month

2_related_ts.csv (data about prices)
- item -- item code
- org -- organizational unit code
- date -- date of sale of the item in format [YYYY]-[MM]-[DD]. First day in a month.
- unit_price -- unit price