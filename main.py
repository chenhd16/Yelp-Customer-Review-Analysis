import nlp_process
import pandas as pd

data = pd.read_csv("/Users/dongni/Desktop/yelp_review_samples_Vancouver.csv")
#
# 只可以跑一次
data['food_pos'] = 0.0
data['service_pos'] = 0.0
data['price_pos'] = 0.0
data['clean_pos'] = 0.0
data['location_pos'] = 0.0

data['food_neg'] = 0.0
data['service_neg'] = 0.0
data['price_neg'] = 0.0
data['clean_neg'] = 0.0
data['location_neg'] = 0.0

#如果想分批跑可以读取你原来存的地方再继续
#data = pd.read_csv("/Users/dongni/Desktop/score.csv")

# (data,20,30) 20是起始行数，30是结尾行数，实际会运行20-29行
ABSA = nlp_process.auto_ABSA(data,20,30)
ABSA.absa_main()

#最后全跑完了跑这个，文件传drive
# toReturn = pd.read_csv("/Users/dongni/Desktop/score.csv")
# toReturn = toReturn.drop(columns = ['text'])
# toReturn.to_csv('/Users/dongni/Desktop/ABSA_Vancouver.csv')