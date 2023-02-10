import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os, json, requests as req, string
import argparse

class CharacterReversalDatagen:
    
    def __init__(self, n, length=10, characters = string.ascii_letters + string.digits):
        self.n = n
        self.length = length
        self.characters = ['\0'] + list(characters)
        self.vocab_size = len(characters)
    
    def gen(self, filename = None):

        output = []
        for i in range(self.n):
            length = np.random.randint(1, self.length)
            x = np.random.choice(list(self.characters[1:]), length)
            y = x[::-1]
            output.append({'x': x, 'y': y})
        
        df = pd.DataFrame(output)
        
        if filename != None:
            df.to_csv(filename, index=False, header=False)
        
        return df

    def embedding(self, df, alphabet = None):

        if alphabet == None:
            alphabet = self.characters
        
        def encode(x:str, target_length = self.length) -> list:
            x = list(x)
            x = [alphabet.index(c) for c in x]
            x = x + [0]*(target_length - len(x))
            return x
        
        df.x = df.x.apply(encode)
        df.y = df.y.apply(encode)
        return df
        
    def decode(self, embedding:list, alphabet = None) -> str:

        if alphabet == None:
            alphabet = self.characters

        x = [alphabet[i] for i in embedding if i != 0]
        x = ''.join(x)
        return x


class ShakespeareDatagen:

    def __init__(self, tokenizer = r"\b", block_size = 50, sampling_rate = 1):
        self.URL = 'https://www.gutenberg.org/files/100/100-0.txt'
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.sampling_rate = sampling_rate
        self.text = req.get(self.URL).text
        self.dictionary = list(set(self.text.split(self.tokenizer)))
        self.vocab_size = len(self.dictionary)
    
    def gen(self, filename = None, trunc: int = None):
        
        text = self.text
        if trunc is not None:
            text = text[:trunc]

        text = text.split(self.tokenizer)
        text = [x for x in text if len(x) > 0]
        output = []
        for i in range(len(text) - self.block_size):

            if np.random.rand() > self.sampling_rate:
                continue
            
            x = text[i:i+self.block_size]
            y = text[i+1:i+self.block_size+1]

            output.append({'x': x, 'y': y})
        
        df = pd.DataFrame(output)
        
        if filename != None:
            df.to_csv(filename, index=False, header=False)
        
        return df

    def embedding(self, df, lexicon = None, token_length = 1):

        if lexicon == None:
            lexicon = self.dictionary
        
        def encode(x:str) -> list:
            x = list(x)
            x = [lexicon.index(c) for c in x]
            return x
        
        df.x = df.x.apply(encode)
        df.y = df.y.apply(encode)
        return df
    
    def decode(self, embedding:list, lexicon = None) -> str:

        if lexicon == None:
            lexicon = self.dictionary

        x = [lexicon[i] for i in embedding]
        x = ''.join(x)
        return x


## testing
if __name__ == "__main__":

    # get argument called name
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='char_reversal')
    parser.add_argument('--debug', type=str, default="on")
    args = parser.parse_args()
    
    if args.debug == "on":
        breakpoint()

    if (args.task == 'char_reversal'):

        # test character reversal
        print('Generating character reversal data...')
        cr = CharacterReversalDatagen(10)
        df = cr.gen()
        embed_df = cr.embedding(df)
        print(embed_df.head())
        print(cr.decode(embed_df.x[0]))


    elif (args.task == 'shakespeare'):

        # test shakespeare
        print('Generating shakespeare data...')
        sh = ShakespeareDatagen()
        df = sh.gen(trunc = 1e4)
        embed_df = sh.embedding(df)
        print(embed_df.head())
        print(sh.decode(embed_df[0]))
    
    else:

        raise ValueError('name not recognized')
