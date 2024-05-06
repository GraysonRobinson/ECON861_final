from sklearn.model_selection import train_test_split
import pandas

dataset = pandas.read_csv("dataset.csv", skipinitialspace=True)

X = dataset["profile"]
y = dataset["stars"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
