import json
import pandas as pd
import pyarrow.parquet as pq
class UtilReadFile:
    @staticmethod
    def read_json(filename) -> pd.DataFrame:
        with open(filename,'r') as file:
         return pd.DataFrame(json.load(file))
        
    
    @staticmethod
    def read_csv_with_pandas(filename) -> pd.DataFrame:
       return pd.read_csv(filename)
    
    @staticmethod
    def read_excel_with_pandas(filename) -> pd.DataFrame:
       return pd.read_excel(filename)
    
    @staticmethod
    def read_parquet(filename)-> pd.DataFrame:
       return pq.read_table(filename).to_pandas()


        
       
    

            