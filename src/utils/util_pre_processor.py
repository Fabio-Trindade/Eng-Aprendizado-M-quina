import pandas as pd


class UtilPreProcessor:
    @staticmethod
    def creat_vocab(texts: list[str]):
        vocab_to_index = {}
        index_to_vocab = {}
        index = 0
        for text in texts:
            if vocab_to_index.get(text) == None:
                vocab_to_index[text] = index
                index_to_vocab[index] = text
                index += 1
        return vocab_to_index,index_to_vocab
    
    @staticmethod
    def get_unique_values_by_columns(df:pd.DataFrame, columns: list[str]):
        values = []
        for column in columns:
            values += df[column].unique().tolist()
        return values
    
    @staticmethod
    def transform_values_in_classes(values: list):
        value_to_class = {}
        class_to_value = {}
        new_list = []
        classs = 0
        for value in values:
            if value_to_class.get(value) == None:
                value_to_class[value] = classs
                class_to_value[classs] = value
                classs += 1
            new_list.append(value_to_class[value])
        return value_to_class, class_to_value, new_list
    
    @staticmethod
    def split_column_data_by_pattern(df:pd.DataFrame,column_name: str, pattern: str):
        return  df[column_name].apply(lambda x: x.split(pattern) )
    
    @staticmethod
    def get_index_vocab(vocab:dict[str,int],string):
        if vocab.get(string) != None:
            return vocab[string]
        else:
            return -1
 