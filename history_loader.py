import yfinance as yf
import pandas as pd

def load_max_history(assets):
  df = None

  for asset in assets:
    data = yf.Ticker(asset).history(period='max', auto_adjust=False)
    columns = pd.MultiIndex.from_product(
      [data.columns, [asset]],
      names=['property', 'asset']
    )
    multi_level_df = pd.DataFrame(data.values, index=data.index, columns=columns)
    # display(multi_level_df.columns.to_numpy())
    df = multi_level_df if df is None else df.join(multi_level_df)

  df = df[sorted(df.columns)]
  return df.dropna()