import json
import numpy as np
import pandas as pd
import aspect_based_sentiment_analysis as absa

class auto_ABSA:
    def __init__(self,dataframe,idx,end):
        self.dataframe = dataframe
        self.idx = idx
        self.end = end
        recognizer = absa.aux_models.BasicPatternRecognizer()
        self.nlp = absa.load('absa/classifier-rest-0.2', pattern_recognizer=recognizer)
        #self.shape = self.dataframe.shape[0]

    def add_column(self):
        self.dataframe['food'] = 0.0
        self.dataframe['service'] = 0.0
        self.dataframe['price'] = 0.0
        self.dataframe['clean'] = 0.0
        self.dataframe['location'] = 0.0

    def slice_text(self,text):
        if len(text) <= 700:
            return [text]
        return [text[:700]] + self.slice_text(text[700:])

    def process_absa(self, length = 100):
        data = self.dataframe['text'][self.idx:self.idx+length]
        for i in range(self.idx,self.idx+length):
            temp = self.slice_text(data[i])
            n = len(temp)
            for j in temp:
                if len(j) != 0:
                    food, service, price, clean, location = self.nlp(j, aspects=['food', 'service', 'price', 'clean','location'])
                    self.dataframe['food_pos'][i] += food.scores[2]
                    self.dataframe['service_pos'][i] += service.scores[2]
                    self.dataframe['price_pos'][i] += price.scores[2]
                    self.dataframe['clean_pos'][i] += clean.scores[2]
                    self.dataframe['location_pos'][i] += location.scores[2]

                    self.dataframe['food_neg'][i] += food.scores[1]
                    self.dataframe['service_neg'][i] += service.scores[1]
                    self.dataframe['price_neg'][i] += price.scores[1]
                    self.dataframe['clean_neg'][i] += clean.scores[1]
                    self.dataframe['location_neg'][i] += location.scores[1]

            self.dataframe['food_pos'][i] = round(self.dataframe['food_pos'][i]/n,2)
            self.dataframe['service_pos'][i] = round(self.dataframe['service_pos'][i]/n,2)
            self.dataframe['price_pos'][i] = round(self.dataframe['price_pos'][i]/n,2)
            self.dataframe['clean_pos'][i] = round(self.dataframe['clean_pos'][i]/n,2)
            self.dataframe['location_pos'][i] = round(self.dataframe['location_pos'][i]/n,2)

            self.dataframe['food_neg'][i] = round(self.dataframe['food_neg'][i] / n, 2)
            self.dataframe['service_neg'][i] = round(self.dataframe['service_neg'][i] / n, 2)
            self.dataframe['price_neg'][i] = round(self.dataframe['price_neg'][i] / n, 2)
            self.dataframe['clean_neg'][i] = round(self.dataframe['clean_neg'][i] / n, 2)
            self.dataframe['location_neg'][i] = round(self.dataframe['location_neg'][i] / n, 2)

        self.onetimeover()

    def onetimeover(self):
        #temp_save = self.idx
        #改这儿
        self.dataframe.to_csv('/Users/dongni/Desktop/score.csv')
        #print(f'Successfully Save {temp_save} to {self.idx} columns')

    def absa_main(self):
        shape = self.end - self.idx
        #shape = self.dataframe.shape[0]
        round = shape // 100
        if round > 0:
            for i in range(round):
                self.process_absa(100)
                temp = self.idx + 99
                print(f'Successfully Save {self.idx} to {temp} columns')
                self.idx += 100
        self.process_absa(shape % 100)
        #temp_end = self.end - 1
        print("That's end")
        #print(f'Successfully Save {self.idx} to {temp_end} columns')
