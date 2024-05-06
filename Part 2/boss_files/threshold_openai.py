import pandas as pd
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

sample_new = pd.read_csv("sample_new_data/sample_new.csv", skipinitialspace=True)

sample_new_transformed = count_vectorize_transformer.transform(sample_new.iloc[:, 1])

prediction_prob = machine.predict_proba(sample_new_transformed)

threshold = 0.15

close_probs_indices = []
distinct_probs_indices = []
for i, probs in enumerate(prediction_prob):
    highest_prob = max(probs)
    second_highest_prob = sorted(probs)[-2]
    if highest_prob - second_highest_prob < threshold:
        close_probs_indices.append(i)
    else:
        distinct_probs_indices.append(i)

print(f"Percentage of samples with close probabilities: {(len(close_probs_indices) / len(sample_new)) * 100:.2f}%")

close_probs_samples = sample_new.iloc[close_probs_indices]
distinct_probs_samples = sample_new.iloc[distinct_probs_indices]

close_probs_samples.to_csv('use_openai.csv', index=False)
distinct_probs_samples.to_csv('use_normal_model.csv', index=False)
