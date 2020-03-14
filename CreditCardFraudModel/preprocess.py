from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class preprocessing(TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, dataset ,*args, **kwargs):
#         dataset = dataset.drop(columns = ["Time"])
        self.stscalers = []
        
        for col in dataset.columns:
            stscaler = StandardScaler()
            stscaler.fit(dataset.loc[:,col].values.reshape(-1,1))
            self.stscalers.append(stscaler)
        
        return self
            
    def transform(self , dataset):
        self.newdataset = pd.DataFrame()
#         dataset = dataset.drop(columns = ["Time"])
        
        for stscaler,col in zip(self.stscalers,dataset.columns):
            feature = stscaler.transform(dataset[col].values.reshape(-1,1))
            self.newdataset[col] = np.reshape(feature, (-1))
        
        return self.newdataset
            
        