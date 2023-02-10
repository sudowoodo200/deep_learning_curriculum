import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os, json, requests as req, string, re
import argparse
from tqdm import tqdm

class CharacterReversalData:
    
    def __init__(self, max_length = 10, min_length = 10, characters = string.digits):
        self.max_length = max_length
        self.min_length = min_length
        self.characters = ['\0'] + list(characters)
        self.vocab_size = len(characters) + 1
    
    def gen(self, N:int = 1e4, filename = None):

        output = []
        for i in range(int(N)):
            length = np.random.randint(self.min_length, self.max_length+1)
            x = np.random.choice(list(self.characters[1:]), length)
            y = x[::-1]
            output.append({'x': x, 'y': y})
        
        df = pd.DataFrame(output)
        
        if filename != None:
            df.to_csv(filename, index=False)
        
        return df

    def encode(self, df, alphabet = None):

        if alphabet == None:
            alphabet = self.characters
        
        def embed(x:str, target_length = self.max_length, trailing = True) -> list:
            x = list(x)
            x = [alphabet.index(c) for c in x]
            
            padding = [0]*(target_length - len(x))
            x = (x + padding) if trailing else (padding + x)
            return x
        
        df.x = df.x.apply(embed, trailing = True)
        df.y = df.y.apply(embed, trailing = False)
        return df
        
    def decode(self, embedding:list, alphabet = None) -> str:

        if alphabet == None:
            alphabet = self.characters

        x = [alphabet[i] for i in embedding if i != 0]
        x = ''.join(x)
        return x


class ShakespeareData:

    def __init__(self, tokenizer = r"\b", block_size = 100):
        self.URL = 'https://www.gutenberg.org/files/100/100-0.txt'
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.text = req.get(self.URL).text[3:]
        self.dictionary = list(set(re.split(self.tokenizer, self.text)))
        self.vocab_size = len(self.dictionary)
    
    def gen(self, filename = None, trunc: int = None):
        
        text = self.text
        text = re.split(self.tokenizer, text)

        if trunc is not None:
            text = text[:int(trunc)]
        text = [x for x in text if len(x) > 0]
        output = []
        for i in range( len(text) - self.block_size+2):
            
            x = text[i:i+self.block_size]
            y = text[i+1:i+self.block_size+1]

            output.append({'x': x, 'y': y})
        
        df = pd.DataFrame(output)
        
        if filename != None:
            df.to_json(filename)
        
        return df

    def encode(self, df, lexicon = None, token_length = 1, filename = None):

        if lexicon == None:
            lexicon = self.dictionary
        
        def embed(x:str) -> list:
            x = list(x)
            x = [lexicon.index(c) for c in x]
            return x
        
        tqdm.pandas(desc="Encoding data...")
        N = len(df)
        df.x = df.x.progress_apply(embed)
        df.y[:N-1] = df.x[1:]
        df.y[N-1] = embed(df.y[N-1])

        if filename != None:  
            df.to_json(filename)

        return df
    
    def decode(self, embedding:list, lexicon = None) -> str:

        if lexicon == None:
            lexicon = self.dictionary

        x = [lexicon[i] for i in embedding]
        x = ''.join(x)
        return x

## Training / Validation split
def split_data(df, test_size = 0.2):
    train, test = train_test_split(df, test_size=test_size)
    return train, test

## Convert to tensor
def to_tensor(df):
    x = torch.tensor(np.array(df.x.tolist()))
    y = torch.tensor(np.array(df.y.tolist()))
    return x, y

## Get batch
def get_batch(x, y, batch_size, i = None):

    # random sampling v.s. sequential sampling
    if i is None:
        index = torch.randint(0, len(x), (batch_size, ))
    else:
        index = torch.arange(i, i+batch_size)

    xb = torch.stack([x[i] for i in index])
    yb = torch.stack([y[i] for i in index])
    return xb, yb

## Decoding Logits to vocab embedding
def decode_logits(logits, lookback_block_size = 1):

    batch_size, block_length, vocab_size = logits.shape

    logits_pred = logits[:, -lookback_block_size : , :] # (batch_size, lookback_block_size, vocab_size)
    probs = F.softmax(logits_pred, dim=-1) # (batch_size, lookback_block_size, vocab_size) where sum over vocab_size = 1
    probs = probs.reshape(batch_size * lookback_block_size, -1) # (batch_size * lookback_block_size, vocab_size
    output = torch.multinomial(probs, num_samples = 1) # (batch_size, lookback_block_size, 1)
    output = output.reshape(batch_size, -1) # (batch_size, lookback_block_size)

    return output

## Decoding vocab embeddings to text
def decode_vocab(x, decoder):
    
    # apply decoder to each row
    # breakpoint()
    x = x.tolist()
    x = [decoder(row) for row in x]
    x = pd.DataFrame(x)
    
    return x


#################### MAIN ####################

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
        cr = CharacterReversalData(10)
        df = cr.gen()
        embed_df = cr.encode(df)
        print(embed_df.head())
        print(cr.decode(embed_df.x[0]))


    elif (args.task == 'shakespeare'):

        # test shakespeare
        print('Generating shakespeare data...')
        sh = ShakespeareData()
        df = sh.gen(trunc = 1e5)
        embed_df = sh.encode(df, filename='./data/shakespeare_embed.json')
        print(embed_df.head())
        print(sh.decode(embed_df.x[0]))
        print(sh.decode(embed_df.y[0]))
    
    else:

        raise ValueError('name not recognized')