# ECON861_final

## Steps for running the program:

# open Part 1 folder
open boss_files
run run_text_analysis_openai.py which will use dataset.csv protected by the .gitignore file and returns dataset_with_new_column
run accuracy_score.py which will return the accuracy of the model which is using OpenAI

### open Part 2 folder
open private_files
run train_machine.py which will use the dataset.csv found in the training_data folder and will return the text_analysis_machine.pickle
open boss_files
run make_prediction.py which will use the trained model from the pickle file and the data from sample_new.csv within the sample_new_data file to return sample_new_with_prediction.csv
run threshold_openai.py which will return use_normal_model.csv and use_openai.csv
run run_text_analysis_openai.py

### open Part 3 folder
open private_files
run logistic_regression.py using dataset.csv
run train_machine.py using dataset.csv which will return text_analysis_pickle
run kfold_template.py
run accuracy_comp.py
open boss_files
run run_cnn.py which will create the pickle file cnn_image_machine.pickle
run make_prediction.ppy using the pickle file and the sample_profile_pictures which will create the file predictions.csv

## Instruction and Explanation for Boss

### Part 1
For Part 1, I created an .env file with my API key and .gitignore file with the datasets and .env contained within but did not push them to Github. run_text_analysis_openai.py is using OpenAI and references the oringinal dataset dataset.csv to predict stars in a new dataset called dataset_with_new_column.csv. This dataset contains the original data along with an added column (openai_predicted_star) which contains the predicted stars from the model. After that, run accuracy_score.py, which will calculate the accuracy of the OpenAI model using the two separate columns and comparing them. After running it a couple times, the model returned: Accuracy Score (Quality of the Programmer): 0.8901869158878505, meaning the OpenAI model correctly predicted the star rating almost 90% of the time. As you adjust the prompt in the model, this percentage might change and should be used as an indicator for how successful the model actually predicts stars. You want this percentage to be as high as possible because as you expand the size of the dataset, the actual number of mispredicted stars will increase with the size of the dataset. 

### Part 2
For Part 2, I also created an .env file with my API key and .gitignore file with the datasets and .env contained within but did not push them to Github. Within the private_files (which the boss will not have access to), I have used a Naive Bayesian model to train a model using the dataset.csv within the training_dataset folder I was provided. This trained model will create a pickle file output named text_analysis_machine.py. Now we can look at what is within the boss_files folder. I have created a program named make_prediction.py which will predict the quality of the programmer. This program will return the dataset sample_new_with_prediction.csv which will have a predicted star and prediction probabilities for how likely each review was to be affiliated with a 1, 2, or 3 score. The column names are: text, reviewtext, prediction, prediction_prob_1, prediction_prob_2, prediction_prob_3. For example, the first review returns values of 1,Laura's lack of commitment and poor system management skills led to a deteriorating infrastructure and numerous unresolved issues.,1,0.9406,0.05934996016563543,0.0. The review had a 94% predicted probability of being a 1 star based on the text review, 5.9% predicted probability of being a 2 star based on the text review, and a 0.0% predicted probability of being a 3 star based on the text review. Thus, using the review, the model predicted the programmer to be a 1 star. However, some of the predicted probabilities were closer to each other so the star was harder to predict. For example, the third review returns values of 3, "Working with Nick is like being trapped in a never-ending nightmare. His code is so bad that it haunts my dreams! I wake up in a cold sweat, screaming at the thought of dealing with his mess again.",1,0.5197,0.45833920097779357,0.0219. As you can see, 51% and 46% are very close. In these situations, it would be more helpful to use OpenAI API to predict the quality stars instead of the model. Given how helpful the predicted probabilities are for determining how close of a decision the star prediction is, I decided using the difference between these predicted probabilities as a threshold would be a good way of determining which reviews need OpenAI (where a prompt and more powerful computing can be used) compared to the simpler model. Therefore, the model threshold_openai.py can be used to determine what the predicted probabilities are for each star rating for each review and then compare the difference between the top two predicted probabilities. ChatGPT was very helpful in this part in regards to combining all the different steps so that the program could seamlessly do everything at once. I ended up using a 0.15 threshold (or in other words the threshold is if the predicted probabilities are within 0.15 percentage points of each other). I settled on this threshold by running the model a couple of times and testing the percentage of reviews that met the threshold.

Threshold Testing
0.1: Percentage of samples meeting the threshold: 14.285714285714285
0.2: Percentage of samples with close probabilities: 28.57%
0.3: Percentage of samples with close probabilities: 57.14%

Thus, I settled on using 1.5 which also returns a percentage of 28.57% of reviews that are close enough to require OpenAI. Depending on how much you want to spend, you can adjust this threshold as you see fit. 

If the difference is above the threshold (or the predicted probability values are far enough apart), then the program automatically stores these distinct_probs_samples in a separate csv file named use_normal_model.csv. From there, it can be determined that those reviews are distinct enough in their predicted probabilities that our normal model can effectively predict them, which can save money. However, if that difference is below a certain threshold (or the values are very close), then it stores these close_probs_samples in a separate csv file named use_openai.csv. From there, those reviews (which were very likely to be either/or), can then be run through the OpenAI program I made in the file, run_text_analysis_openai.py to give a more accurate answer as to their predicted star quality rating. When run, this program will use OpenAI and return a dataset named dataset_using_openai.csv which will contain the predictions. 

### Part 3
For Part 3, within the private_files (which again the boss will not have access to), I have created a logistic_regression.py model and a train_machine.py (Naive Bayesian model) to train the model. I also created a kfold_template.py model to validate. In addition, if you run the accuracy_comp.py model, it will compare the results of the logistic regression and Naive Bayesian model to determine which one should be used. The results are below:

python3 accuracy_comp.py
Naive Bayesian Model Performance:
Accuracy: 0.7906976744186046
Precision: 0.8007818799793568
Recall: 0.7906976744186046
F1 Score: 0.7717159093067799
ROC AUC Score: 0.8440896096903335

Logistic Regression Model Performance:
Accuracy: 0.7906976744186046
Precision: 0.8007818799793568
Recall: 0.7906976744186046
F1 Score: 0.7717159093067799
ROC AUC Score: 0.8440896096903335

Best Model: Naive Bayesian

Therefore, the higher scores and the fact that the model so easily outputs a pickle file led to my decision to use the Naive Bayesian model (named train_machine.py in my folder). From there, enter the boss_files folder and run the run_cnn.py program, which will be saved into a pickle file named cnn_image_machine.pickle. After that, run the make_prediction.py program using the pickle file to actually create an output file named predictions.csv, which accurately predicts the images with a quality rating (0 for building, 1 for dog, 2 for face). All inall, Part 3 is training a model to predict stars based on dataset.csv, saving that model in a pickle file, training a model to classify the profile picture type and save in a file, and predict the quality of the programmer based on the profile pictures. This step took me a long time and I learned a lot from it - image recognition is really neat!


## Datasets and Hidden Files
I did not upload the datasets from each step. I also did not upload all of the .env files as they contain a private API key.

## Sources:
I used notes from class, Github, and ChatGPT to help with the code. I used the class notes and Github to help get started and work on structure, as well as when I didn’t understand a python concept, and ChatGPT to help proofread, combine, debug, and iterate/improve my code. 
