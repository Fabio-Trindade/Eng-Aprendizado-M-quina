import pandas as pd

from src.utils.util_pre_processor import UtilPreProcessor


class PreProcessorSpaceship:
    def __init__(self, csv_train_filename: str):
        self.df = pd.read_csv(csv_train_filename)
        self.pre_processed_df = None
        self.vocab = None
    
    def get_columns_name(self):
        return self.df.columns

    def get_words_by_column(sel,df,column_name):
        words = set()
        for words_list in df[column_name]:
            for word in words_list:
                words.add(word)
        return list(words)


    def pre_process(self):
        if self.pre_processed_df != None:
            return self.pre_processed_df, self.vocab

        pre_processed_df = self.df.copy() 

        dict_max_len = {}

        other_columns_to_be_processed = ['PassengerId','Cabin','Name']
        columns_to_be_processed = ['HomePlanet','Destination']
        boolean_columns = ["CryoSleep", "VIP"]
        remain_columns =["Age","RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]
        
        for column in remain_columns:
            pre_processed_df[column] = pre_processed_df[column].fillna(-1) 
        
        for column in boolean_columns + other_columns_to_be_processed + columns_to_be_processed + ["Transported"]:
            pre_processed_df[column] = pre_processed_df[column].astype(str)

        for column in remain_columns:
            pre_processed_df[column] = pre_processed_df[column].astype(float)
        
        pre_processed_df["PassengerId"] = UtilPreProcessor.split_column_data_by_pattern(pre_processed_df,"PassengerId","_")
        pre_processed_df["Cabin"] = UtilPreProcessor.split_column_data_by_pattern(pre_processed_df,"Cabin","/")
        pre_processed_df["Name"] = UtilPreProcessor.split_column_data_by_pattern(pre_processed_df,"Name"," ")        

        passenger_id_words = self.get_words_by_column(pre_processed_df,"PassengerId")
        cabin_words = self.get_words_by_column(pre_processed_df,"Cabin")
        name_words = self.get_words_by_column(pre_processed_df,"Name")
        words = UtilPreProcessor.get_unique_values_by_columns(pre_processed_df,columns_to_be_processed + boolean_columns) + columns_to_be_processed  \
                                                    + boolean_columns + passenger_id_words + cabin_words + name_words + \
        [f"{i}{j}{k}{l}" for i in range(10) for j in range(10) for k in range(10) for l in range(10)] + [str(i) for i in range(10**4)] + ['unknown']
        vocab_to_index, index_to_vocab = UtilPreProcessor.creat_vocab(words)
        
        for column in other_columns_to_be_processed:
            listt = pre_processed_df[column].to_list()
            max_len = max(len(sublist) for sublist in listt)
            dict_max_len[column] = max_len

        for column in other_columns_to_be_processed:
            pre_processed_df[column] = pre_processed_df[column].apply(lambda x: [vocab_to_index[vocab] for vocab in x] if len(x) == dict_max_len[column] else [vocab_to_index['nan'] for _ in range(dict_max_len[column])])

        for column in columns_to_be_processed:
            pre_processed_df[column] = pre_processed_df[column].apply(lambda x: vocab_to_index[x])

        for column in boolean_columns:
            pre_processed_df[column] = pre_processed_df[column].apply(lambda x: [vocab_to_index[x],vocab_to_index[column]])

        pre_processed_df['Transported'] = pre_processed_df['Transported'].apply(lambda x: vocab_to_index[x])
        pre_processed_data = []
        
        for i,row in enumerate(pre_processed_df.itertuples(index=False)):
            pre_processed_data.append([])
            for data in (row):
                if data.__class__ is list:
                    pre_processed_data[i] += data
                else:
                    pre_processed_data[i].append(data)

        self.vocab = [vocab_to_index,index_to_vocab]
        self.pre_processed_df = pre_processed_df
        return pre_processed_data, self.vocab
    
    def pre_process_test(self,csv_filename):
        df_test = pd.read_csv(csv_filename)

        pre_processed_df = df_test.copy() 

        dict_max_len = {}

        other_columns_to_be_processed = ['PassengerId','Cabin','Name']
        columns_to_be_processed = ['HomePlanet','Destination']
        boolean_columns = ["CryoSleep", "VIP"]
        remain_columns =["Age","RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]
        
        for column in remain_columns:
            pre_processed_df[column] = pre_processed_df[column].fillna(-1) 
        
        for column in boolean_columns + other_columns_to_be_processed + columns_to_be_processed:
            pre_processed_df[column] = pre_processed_df[column].astype(str)

        for column in remain_columns:
            pre_processed_df[column] = pre_processed_df[column].astype(float)
        
        pre_processed_df["PassengerId"] = UtilPreProcessor.split_column_data_by_pattern(pre_processed_df,"PassengerId","_")
        pre_processed_df["Cabin"] = UtilPreProcessor.split_column_data_by_pattern(pre_processed_df,"Cabin","/")
        pre_processed_df["Name"] = UtilPreProcessor.split_column_data_by_pattern(pre_processed_df,"Name"," ")        

        
        for column in other_columns_to_be_processed:
            listt = pre_processed_df[column].to_list()
            max_len = max(len(sublist) for sublist in listt)
            dict_max_len[column] = max_len

        for column in other_columns_to_be_processed:
            pre_processed_df[column] = pre_processed_df[column].apply(lambda x: [self.vocab[0][vocab] if self.vocab[0].get(vocab) != None else self.vocab[0]['unknown']  for vocab in x] if len(x) == dict_max_len[column] else [self.vocab[0]['nan'] for _ in range(dict_max_len[column])])

        for column in columns_to_be_processed:
            pre_processed_df[column] = pre_processed_df[column].apply(lambda x:self.vocab[0][x] if self.vocab[0].get(x) != None else self.vocab[0]['unknown'] )

        for column in boolean_columns:
            pre_processed_df[column] = pre_processed_df[column].apply(lambda x: [self.vocab[0][x] if self.vocab[0].get(x) != None else self.vocab[0]['unknown'] ,self.vocab[0][column]])

        pre_processed_data = []
        
        for i,row in enumerate(pre_processed_df.itertuples(index=False)):
            pre_processed_data.append([])
            for data in (row):
                if data.__class__ is list:
                    pre_processed_data[i] += data
                else:
                    pre_processed_data[i].append(data)

        return pre_processed_data



