import pandas as pd
import os
import io
import random


def getDataset():
    data = pd.DataFrame()
    try:
        data = pd.read_csv('dataset.csv', encoding='iso-8859-9')
    except:
        dataset = []
        for Root, Dirs, Files in os.walk("3000tweet/raw_texts"):
            for di in Dirs:
                for root, dirs, files in os.walk(f"3000tweet/raw_texts/{di}"):
                    for file in files:
                        sub_data = []
                        with io.open(f"3000tweet/raw_texts/{di}/"+file, 'r', encoding='iso-8859-9') as f:
                            text = f.read()
                            sub_data.append(text)
                            sub_data.append(str(int(di)-1))
                        dataset.append(sub_data)
            random.shuffle(dataset)
        data = pd.DataFrame(dataset, columns=['Sentence', 'Sentiment'])
        data.to_csv('dataset.csv', index=False, encoding='iso-8859-9')
        print("No csv file was found!, new file was created :)")
    data.dropna(inplace=True)
    return data
