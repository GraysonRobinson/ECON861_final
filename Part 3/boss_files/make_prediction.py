import pickle
import pandas
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy
import os

machine = pickle.load(open('cnn_image_machine.pickle', 'rb'))

csv_path = 'sample_new_data/sample_new.csv'
df = pandas.read_csv(csv_path, skipinitialspace=True)

image_dir = 'sample_new_data/sample_profile_pictures'

new_data_generator = ImageDataGenerator(rescale=1./255)

new_data = new_data_generator.flow_from_dataframe(
    dataframe=df,
    directory=image_dir,
    x_col='profile_picture',  
    y_col=None,  
    target_size=(50, 50),
    batch_size=1,
    class_mode=None,  
    shuffle=False  
)

predictions = numpy.argmax(machine.predict(new_data), axis=1)

results = pandas.DataFrame({'image': new_data.filenames, 'prediction': predictions})

results.to_csv('predictions.csv', index=False)
