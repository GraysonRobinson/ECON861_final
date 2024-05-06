import pandas
import dill as pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import nltk

nltk.download("stopwords")
nltk.download("wordnet")

def pre_processing(text, stop_words, lemmatizer):
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [lemmatizer.lemmatize(word.lower()) for word in text.split() if word.lower() not in stop_words]
    return " ".join(tokens)

with open("text_analysis_machine.pickle", "rb") as f:
    machine = pickle.load(f)
    count_vectorize_transformer = pickle.load(f)
    lemmatizer = pickle.load(f)
    stop_words = pickle.load(f)

sample_new = pandas.read_csv("sample_new_data/sample_new.csv", skipinitialspace=True)

sample_new_transformed = count_vectorize_transformer.transform(sample_new.iloc[:, 1])

prediction = machine.predict(sample_new_transformed)
prediction_prob = machine.predict_proba(sample_new_transformed)

print(prediction)
print(prediction_prob)

sample_new['prediction'] = prediction
prediction_prob_dataframe = pandas.DataFrame(prediction_prob)
prediction_prob_dataframe = prediction_prob_dataframe.rename(columns={
    0: "prediction_prob_1",
    1: "prediction_prob_2",
    2: "prediction_prob_3"
})
sample_new = pandas.concat([sample_new, prediction_prob_dataframe], axis=1)
sample_new = sample_new.rename(columns={sample_new.columns[0]: "text"})
sample_new['prediction'] = sample_new['prediction'].astype(int)
sample_new['prediction_prob_1'] = round(sample_new['prediction_prob_1'], 4)
sample_new['prediction_prob_3'] = round(sample_new['prediction_prob_3'], 4)

sample_new.to_csv("sample_new_with_prediction.csv", index=False)
