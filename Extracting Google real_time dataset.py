from pytrends.request import TrendReq
import pandas as pd

pytrends = TrendReq(hl='en-US', tz=360)
keywords = ['refugee', 'migration', 'asylum', 'border crossing']
pytrends.build_payload(keywords, cat=0, timeframe='now 1-d', geo='')

data = pytrends.interest_over_time()
data = data.reset_index()
data.to_csv("google_trends.csv", index=False)
