import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import dill as pickle

nltk.download("stopwords")
nltk.download("wordnet")

dataset = pandas.read_csv("dataset.csv")

dataset = dataset[0:3000]
dataset = dataset[(dataset['stars'] == 1) | (dataset['stars'] == 2) | (dataset['stars'] == 3)]

print(dataset)

data = dataset["profile"]
target = dataset["stars"]

lemmatizer = WordNetLemmatizer()

def pre_processing(text, stopwords, lemmatizer):
    text_processed = "".join([char for char in text if char not in "!,.?;:\"'"])
    text_processed = text_processed.split()
    result = []
    for word in text_processed:
        word_processed = word.lower()
        if word_processed not in stopwords:
            word_processed = lemmatizer.lemmatize(word_processed)
            result.append(word_processed)
    return result

stop_words = set(stopwords.words("english"))

count_vectorize_transformer = CountVectorizer(analyzer=lambda x: pre_processing(x, stop_words, lemmatizer)).fit(data)

data_transformed = count_vectorize_transformer.transform(data)

machine = MultinomialNB()
machine.fit(data_transformed, target)

with open("text_analysis_machine.pickle", "wb") as f:
    pickle.dump(machine, f)
    pickle.dump(count_vectorize_transformer, f)
    pickle.dump(lemmatizer, f)
    pickle.dump(stop_words, f)
