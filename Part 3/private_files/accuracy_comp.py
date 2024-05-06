import pandas as pandas
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

dataset = pandas.read_csv("dataset.csv")

X = dataset["profile"]
y = dataset["stars"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

nb_model = MultinomialNB()
nb_model.fit(X_train_vectorized, y_train)
nb_predictions_prob = nb_model.predict_proba(X_test_vectorized)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_vectorized, y_train)
lr_predictions_prob = lr_model.predict_proba(X_test_vectorized)

nb_accuracy = accuracy_score(y_test, nb_model.predict(X_test_vectorized))
lr_accuracy = accuracy_score(y_test, lr_model.predict(X_test_vectorized))

nb_precision = precision_score(y_test, nb_model.predict(X_test_vectorized), average='weighted')
lr_precision = precision_score(y_test, lr_model.predict(X_test_vectorized), average='weighted')

nb_recall = recall_score(y_test, nb_model.predict(X_test_vectorized), average='weighted')
lr_recall = recall_score(y_test, lr_model.predict(X_test_vectorized), average='weighted')

nb_f1 = f1_score(y_test, nb_model.predict(X_test_vectorized), average='weighted')
lr_f1 = f1_score(y_test, lr_model.predict(X_test_vectorized), average='weighted')

nb_roc_auc = roc_auc_score(y_test, nb_predictions_prob, average='weighted', multi_class='ovr')
lr_roc_auc = roc_auc_score(y_test, lr_predictions_prob, average='weighted', multi_class='ovr')

print("Naive Bayesian Model Performance:")
print("Accuracy:", nb_accuracy)
print("Precision:", nb_precision)
print("Recall:", nb_recall)
print("F1 Score:", nb_f1)
print("ROC AUC Score:", nb_roc_auc)
print()

print("Logistic Regression Model Performance:")
print("Accuracy:", lr_accuracy)
print("Precision:", lr_precision)
print("Recall:", lr_recall)
print("F1 Score:", lr_f1)
print("ROC AUC Score:", lr_roc_auc)
print()

best_model = max([(nb_accuracy, "Naive Bayesian"), (lr_accuracy, "Logistic Regression")])
print("Best Model:", best_model[1])
