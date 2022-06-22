import os
import pickle
from math import ceil
import numpy as np
import pandas as pd
from typing import Optional, Any
from tensorflow import expand_dims
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from data_cleansing.preprocessing_and_splitting_cat import save_pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer

class NLP_Sequencer(Sequence):
    
    def __init__(self,data_path:str, maxlen:int, batch_size:int=1, tokenizer_path:Optional[str]="E:\\Austria\\the-ventury-nlp\\data\\objects\\tokenizer.pickle",encoder_path:Optional[str]="E:\\Austria\\the-ventury-nlp\\data\\objects\\labelencoder.pickle"):
        """NLP_Sequencer initialization with parameters

        Args:
            data_path (str): Path to data split
            maxlen (int): Maximum length of sentence
            batch_size (int, optional): Batch size during the training. Defaults to 1.
            tokenizer_path (str, optional): Path to tokenizer object. Defaults to "E:\Austria\the-ventury-nlp\data\objects\tokenizer.pickle".
            encoder_path (str, optional): Path to LabelBinarizer object. Defaults to "E:\Austria\the-ventury-nlp\data\objects\labelencoder.pickle".
        """        
        data = NLP_Sequencer.read_pickle(data_path).dropna()
        self.input_body = data["body"]
        
        self.batch_size = batch_size
        self.maxlen = maxlen

        self.tokenizer = NLP_Sequencer.read_pickle(tokenizer_path)
        self.tokenize_inputs()
        if "classification" in data_path:
            labelencoder= NLP_Sequencer.read_pickle(encoder_path) 
            self.output_category = data["category"]
            self.transformed_outputs = labelencoder.transform(self.output_category)
        elif "regression" in data_path:
            self.transformed_outputs = data["score"].values.reshape(-1,1)[:20000,:]
            self.transformed_inputs = self.transformed_inputs[:20000,:]
        self.output_shape = self.transformed_outputs.shape[-1]
        
        
        
    @staticmethod
    def read_pickle(filepath:str)->Any:
        """Read pickle files

        Args:
            filepath (str): Filepath to pickle file

        Returns:
            Any: Red pickle file
        """        
        with open(filepath, 'rb') as handle:
            file = pickle.load(handle)
        return file
    
    
    @staticmethod
    def create_tokenizer(data:pd.DataFrame, filename:str="tokenizer.pickle", num_words:int=20000, oov_token:str='<UNK>'):
        """Create tokenizer

        Args:
            data (pd.DataFrame): Data which needed to be tokenized. This will be the vocab of the object.
            filename (str, optional): Filename of the Tokenizer. Defaults to "tokenizer.pickle".
            num_words (int, optional): First <num_word> in the data. This wil. Defaults to 20000.
            oov_token (str, optional): If a word unknown later it will be replaced with this. Defaults to '<UNK>'.
        """               
        tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
        tokenizer.fit_on_texts(data)
        save_pickle(os.path.join("E:\\Austria\\the-ventury-nlp\\data\\objects\\",filename), tokenizer)
    
    @staticmethod
    def create_encoder(data:pd.DataFrame, filename:str="labelencoder.pickle"):
        """Create LabelBinarizer to handle categorical data.

        Args:
            data (pd.DataFrame): Data which needed to be tokenizer.
            filename (str, optional): Filename of the Encoder. Defaults to "labelencoder.pickle".
        """            
        label_as_binary = LabelBinarizer()
        label_as_binary.fit(data)
        save_pickle(os.path.join("E:\\Austria\\the-ventury-nlp\\data\\objects\\",filename), label_as_binary)
        
    
    @staticmethod
    def expand(arr:np.ndarray,axis:list)->np.ndarray:
        """Expand dims recursively.

        Args:
            arr (np.ndarray): Array which needed to be expanded.
            axis (list): List of axis where expand happens.

        Returns:
            np.ndarray: Expanded array.
        """        
        for idx, ax in enumerate(axis):
            arr=expand_dims(arr,axis=ax)
        return arr
    
    @staticmethod
    def inverse_labeling(labelencoder:LabelBinarizer, data: np.ndarray)->np.ndarray:
        """Inverse labeling the predicted values.

        Args:
            labelencoder (LabelBinarizer): Fited LabelBinarizer.
            data (np.ndarray): Data which needed to be inversed.

        Returns:
            np.ndarray: Inverted data.
        """        
        return labelencoder.inverse_transform(data)
        
    
    def tokenize_inputs(self,pad_type:str="post",trunc_type:str="post"):
        """Tokenize preprocessed text with a fited Tokenizer

        Args:
            pad_type (str, optional): Padding type. Defaults to "post".
            trunc_type (str, optional): Truncate type. Defaults to "post".
        """        
        train_sequences = self.tokenizer.texts_to_sequences(self.input_body)
        self.transformed_inputs = pad_sequences(train_sequences, padding=pad_type, truncating=trunc_type, maxlen=self.maxlen)
        
    
    def __len__(self)->int:
        """Returns the number of iterations based on the number of records and batch size.

        Returns:
            int: Number of iterations.
        """        
        return ceil(len(self.transformed_inputs)/self.batch_size)
    
    def __getitem__(self, idx:int):
        """Generate inputs and outputs for the training based on index and batch size.

        Args:
            idx (int): Index

        Returns:
            tuple: Generated examples
        """        
        inputs = self.transformed_inputs[idx*self.batch_size:(idx+1)*self.batch_size]
        outputs = self.transformed_outputs[idx*self.batch_size:(idx+1)*self.batch_size]
        
        return inputs, outputs

        
        
        