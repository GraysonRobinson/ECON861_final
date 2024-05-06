import pandas
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

def simple_call(prompt):
    completions = client.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=[
            {"role": "user", "content": prompt},
        ], 
        max_tokens=200,
        temperature=0.1,
        top_p=1
    )
    response = completions.choices[0].message.content
    predicted_star = ''.join(filter(str.isdigit, response))
    return predicted_star

dataset = pandas.read_csv('dataset.csv', skipinitialspace=True)

dataset['openai_predicted_star'] = dataset['reviewtext'].apply(lambda x: simple_call("In a scale of 1 to 3, how would you rate the quality of the following programmer based on the review: \"" + x + "\", answer in one number."))

dataset.to_csv('dataset_with_new_column.csv', index=False)
