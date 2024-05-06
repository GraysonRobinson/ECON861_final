import pandas
from sklearn.metrics import accuracy_score

dataset = pandas.read_csv('dataset_with_new_column.csv')

stars_ground_truth = dataset['stars']
quality_predicted = dataset['openai_predicted_star']

accuracy = accuracy_score(stars_ground_truth, quality_predicted)

print("Accuracy Score (Quality of the Programmer):", accuracy)
