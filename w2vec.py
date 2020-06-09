import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Conv1D, MaxPool1D, GlobalMaxPool1D, Activation, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import re
import nltk
from nltk.corpus import stopwords
from sklearn import preprocessing
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from os import listdir
from os.path import isfile, join
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency


import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

#--------------------------------KAYNAKLAR-------------------------------------------
#Kaynak: https://www.kaggle.com/nasdi1/98-accuracy-word2vec-cnn-text-classification/
#Kaynak: https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial
#Kaynak: https://github.com/turuncanil/Sentiment-Tweet-Analysis/blob/master/Sentiment%20Analysis/3k.ipynb
#Kaynak: https://womaneng.com/one-hot-encoding-nedir-nasil-yapilir/
#Kaynak: https://keras.io/api/layers/activations/
#Kaynak: https://keras.io/api/optimizers/
#Kaynak: https://www.kaggle.com/guichristmann/lstm-classification-model-with-word2vec
#-----------------------------------------------------------------------------------

path1 = "3000tweet/raw_texts/1"
path2 = "3000tweet/raw_texts/2"
path3 = "3000tweet/raw_texts/3"


files1 = [f for f in listdir(path1) if isfile(join(path1, f))]
files2 = [f for f in listdir(path2) if isfile(join(path2, f))]
files3 = [f for f in listdir(path3) if isfile(join(path3, f))]
totalFileCount = len(files1) + len(files2) + len(files3)
print("Positive tweet count:",len(files1))
print("Negative tweet count:",len(files2))
print("Neutral tweet count:",len(files3))
print("Total tweet count:",len(files1) + len(files2) + len(files3))


df = pd.DataFrame(columns=['Sentiment','SentimentText'],index=range(totalFileCount))

for idx, file in enumerate(files1):
    file = open( path1 + "/" + file, "r", encoding="cp1254") 
    sentimentDict = {}
    sentimentDict["Sentiment"] = 1
    sentimentDict["SentimentText"] =  file.read() 
    df.iloc[idx] = sentimentDict
    
for idx, file in enumerate(files2):
    file = open( path2 + "/" + file, "r", encoding="cp1254") 
    sentimentDict = {}
    sentimentDict["Sentiment"] = 2
    sentimentDict["SentimentText"] =  file.read() 
    df.iloc[len(files1) + idx] = sentimentDict
    
for idx, file in enumerate(files3):
    file = open( path3 + "/" + file, "r", encoding="cp1254") 
    sentimentDict = {}
    sentimentDict["Sentiment"] = 3
    sentimentDict["SentimentText"] =  file.read() 
    df.iloc[len(files1) + len(files2) + idx] = sentimentDict

#Null değerleri silme işlemi    
df.isnull().sum()
df = df.dropna().reset_index(drop=True)
df.isnull().sum()
#Gereksiz kelime, boşluk, işaretleri silmek için
nltk.download('stopwords')
stop_word_list = stopwords.words('turkish')
stop_word_list

def preprocess_text(sen):
    
    # Sayıları Silme İşlemi
    sentence = re.sub('[\d\s]', ' ', str(sen))

    # Noktalama İşaretlerini Silme İşlemi
    sentence = re.sub('[^\w\s]', ' ', str(sentence))
    
    # Tek Karakterleri Silme İşlemi
    sentence = re.sub(r"\b[\w\s]\b", ' ',str(sentence))
    
    # Birden Çok Boşluğu Silme İşlemi
    sentence = re.sub(r'\s+', ' ', sentence)
        
    # Engellenecek Kelimeleri Silme İşlemi
    WPT = nltk.WordPunctTokenizer()
    tokens = WPT.tokenize(sentence)
    filtered_tokens = [token for token in tokens if token not in stop_word_list]
    single_doc = ' '.join(filtered_tokens)
    
    # Tüm Harfler Küçük Harfe Dönüştürülüyor
    return single_doc.lower()

#X ve Y leri ayırma işlemi
x = df['SentimentText']
y = df['Sentiment']

x = x.apply(preprocess_text)
pd.DataFrame(data=x)

#Kelimeleri ayırma ve kelime haznesi oluşturmak için
words = []
for i in x:
    words.append(i.split())

#Word2Vec modelimizi oluşturalım.
##min_count=Parametre değerinden daha düşük frekanstaki tüm kelimeler yok sayılır(2,100)
#window=Bir cümle içindeki mevcut ve tahmin edilen kelime arasındaki maksimum mesafedir(2,10)
#size=Kelime vektörlerinin boyutudur.(50,300)
#sample=Hangi yüksek frekanslı kelimelerin rastgele altörnekleneceğini yapılandırma eşiği. Son derece etkileyicidir.(0, 1e-5)
#alpha=Başlangıç öğrenme oranıdır.(0, 1e-5)
#min_alpha=Eğitim ilerledikçe learning rate doğrusal olarak min_alfa'ya düşecektir.
#negative=negatif örnekleme kullanılacak, negatif için int kaç "gürültü kelime" boğulmak gerektiğini belirtir. 0 olarak ayarlanırsa, negatif örnekleme kullanılmaz(5, 20)
#workers=Modeli eğitmek için bu birçok işçi iş parçacığını kullanın (= çok çekirdekli makinelerle daha hızlı eğitim)    
word2vec_model = Word2Vec(words, size = 200, window = 5, min_count = 1, workers = 16, sample=0.01,  min_alpha=0.0001, negative=0)

#Veri kümemizdeki Textleri sayılara dönüştürmek için
token = Tokenizer()
token.fit_on_texts(x)
x = token.texts_to_sequences(x)
x = pad_sequences(x)

#Eğitilecek olan verilerimizi ölçeklendirelim.
scaler = StandardScaler()
x = scaler.fit_transform(x)

#Y verilerimize one hot encoding işlemi uyguluyoruz.
#One hot encoding: One Hot Encoding, kategorik değişkenlerin ikili (binary) olarak temsil edilmesi anlamına gelmektedir.
#Bu işlem, ağın model için problemi daha kolay hale getirmesine yardımcı olabilir.
encode = preprocessing.LabelEncoder()
y = encode.fit_transform(y)
y = to_categorical(y)

#Train ve Test işlemleri için
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5, random_state = 0)

print('x Train : ' + str(x_train.shape))
print('x Test : ' + str(x_test.shape))
print('y Train : ' + str(y_train.shape))
print('y Test : ' + str(y_test.shape))

#Katmanları oluşturuyoruz.
model = Sequential()
model.add(word2vec_model.wv.get_keras_embedding(True))
model.add(LSTM(units=128))
#3 adet çıkış verimiz olduğu için
model.add(Dense(3, activation='sigmoid'))

#Modelimizi eğitiyoruz
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
##loss fonksiyonu, gerçek değer ile tahmin edilen değer arasındaki hatayı ifade eden metrik.
egitim = model.fit(x_train, y_train, batch_size=128, epochs=40, validation_data=(x_test, y_test))
#Batch size=parametre güncellemesinin gerçekleştiği sırada ağa verilen alt örneklerin sayısıdır.
#Daha doğru gradyan değerinin hesaplanmasını sağlamaktadır. Bu durum da linerizasyonu azaltmaktadır.
#Batch size genelde 64ile 512 arasında 2'nin katı olan değerlerden belirleniyor.

scores = model.evaluate(x_test, y_test, verbose = 0)
print('Test score:', scores[0]*100)
print('Test accuracy:', scores[1]*100)

#Grafik üzerinde loss ve accuracy değerlerini görelim
fig, ax = plt.subplots(2, 1, figsize=(12,5))
ax[0].plot(egitim.history['loss'], color='g', label="Training loss")
ax[0].plot(egitim.history['val_loss'], color='b', label="Validation loss",axes =ax[0])
ax[0].grid(color='black', linestyle='-', linewidth=0.25)
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(egitim.history['accuracy'], color='g', label="Training accuracy")
ax[1].plot(egitim.history['val_accuracy'], color='b',label="Validation accuracy")
ax[1].grid(color='black', linestyle='-', linewidth=0.25)
legend = ax[1].legend(loc='best', shadow=True)