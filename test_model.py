

import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import seaborn as sb

class EvaluateModel():
  """
    A class to make evalutions on any test dataset for classification task.
    Given, model_file : Path to trained model
           test_folder : Path to test images folder
           target_size : Size to which images need to be resized
           class_mode : whether binary or categorical
           batch_size : default 16

    Methods :
    1. preprocess_data : Creates a dataset generator with rescaled values.
          Args : None
          Returns : Data generator object

    2. make_predictions:Loads the model and makes predictions
          Args : Data generator object to test on
          Returns : Predictions
    3. get_classification_report : Print and save the classification report
          Args : True labels, Predicted labels, file_path to save csv file
          Returns : None, prints report
    4. get_confusion_matrix : Display and save confusion matrix
          Args : True labels, Predicted labels, file_path to save image
          Returns : Confusion Matrix
    5. get_metrics : Obtain the TP, TN, FP, FN values where Resume class is positive.
          Args : confusion matrix for the data
          Returns : a dictionary of metrics values

  """
  def __init__(self, model_file, test_folder, target_size=(224, 224), batch_size=16):
    self.model_file = model_file
    self.test_folder = test_folder
    self.target_size = target_size

    self.batch_size = batch_size

  def preprocess_load_data(self):
    images = []
    labels = []

    # Iterate through the directory structure
    for root, dirs, files in os.walk(self.test_folder):
        for dir_name in dirs:
            label = 0 if dir_name == 'Non-resume' else 1

            # Iterate through images in each subdirectory
            for filename in os.listdir(os.path.join(root, dir_name)):
                img_path = os.path.join(root, dir_name, filename)

                # Load and preprocess the image
                img = image.load_img(img_path, target_size=(224, 224))
                x = image.img_to_array(img)
                x = x / 255.0  # Normalize pixel values to the range [0, 1]

                images.append(x)
                labels.append(label)

    # Convert lists to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels

  def make_predictions(self, images):
    # Load the model
    model=load_model(self.model_file)

    # Make Predictions
    predictions = model.predict(images, batch_size = self.batch_size, verbose=1)

    # Convert the predictions to 0 and 1 based on threshold 0.5
    predicted_labels = []
    for i in predictions:
      if i[0] >= 0.5:
        predicted_labels.append(1)

      else:
        predicted_labels.append(0)

    return predicted_labels

  def get_classification_report(self, true_labels, predicted_labels, file_name='classification_report.csv'):
    # Print the classification report
    print(classification_report(true_labels, predicted_labels))

    # Save the report as a csv file
    report = classification_report(true_labels, predicted_labels, target_names=['Non-resume', 'Resume'], output_dict=True)
    report_df = pd.DataFrame(report)

    report_df.to_csv(file_name)

  def get_confusion_matrix(self, true_labels, predicted_labels, file_name='confusion_matrix.jpg'):
    # Obtain the confusion matrix
    matrix = confusion_matrix(true_labels, predicted_labels)

    # Convert the matrix to a data frame with rows and indexes defined
    matrix_df = pd.DataFrame(matrix, index=['Non-resume', 'Resume'], columns=['Non-resume', 'Resume'])

    # Plot a heatmap for the dataframe
    ax = sb.heatmap(matrix_df, annot=True, cmap='Blues')
    plt.savefig(file_name)

    plt.show(block=True)

    return matrix

  def get_metrics(self, confusion_matrix):
    # Obtain the metrics from the confusion matrix
    tp = confusion_matrix[1, 1]
    tn = confusion_matrix[0, 0]
    fp = confusion_matrix[1, 0]
    fn = confusion_matrix[0, 1]

    # Create the dictionary of metrics
    metrics = {'True Positives':tp, 'True Negatives':tn, 'False Positives': fp, 'False Negatives':fn}
    return metrics

# Code to run in Command line
if __name__=="__main__":
  import sys

  model_file = sys.argv[1]
  test_folder = sys.argv[2]
  target_size = sys.argv[3]
  batch_size = sys.argv[4]

  model_test = EvaluateModel(model_file=model_file, test_folder=test_folder, target_size=target_size, batch_size=batch_size)
  images, labels = model_test.preprocess_load_data()
  pred = model_test.make_predictions(images)
  model_test.get_classification_report(labels, pred)
  cm = model_test.get_confusion_matrix(labels, pred, verbose=False)
  model_test.get_metrics(cm)

