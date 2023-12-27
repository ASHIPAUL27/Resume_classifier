# Resume_classifier
**Building a Resume Image Classifier using InceptionV3 model.** 

I have built a model that classifies resume images from non-resume documents efficiently by only taking into consideration the visual features of the documents. We can observe that a resume stands out from other documents in the way that it has a more section-wise structure, where it follows a particular font throughout, has a constant color theme and will contain just one image of the candidate only, compared to other documents with multiple fonts or colors, logos, images and other features. This model has learnt these features effectively and will be able to classify resumes from a bunch of other document images.

The model used here is the **Inceptionv3** model from the keras library, optimizer Adam with learning rate 0.001 and 100 epochs are completed with the binary crossentropy loss.

The trained model is saved as 'model_checkpoint.h5' and can be used to do evaluations on any document images.

# Data Set:
The Data set is collected by scraping images from google using the python library ‘pygoogle_image’. We collected 579 resume images with varying characteristics and 611 non-resume images which include the following categories : articles, aadhar cards, certificates, emails, handwritten notes, memos, news articles, posters, registration forms and scientific reports. The dataset was split as 80% train, 10% validation and 10% test.

![image](https://github.com/ASHIPAUL27/Resume_classifier/assets/152466355/26c9a1fe-595d-4f21-b0ed-a86c67d89fff)

![image](https://github.com/ASHIPAUL27/Resume_classifier/assets/152466355/511ab9db-a038-463f-84e5-ae1874e617db)



# Testing images

I have provided a python script that has a class capable of creating a object that makes evaluations on any dataset with the trained model. It can provide us with predictions for the data, give us the classification report, confusion matrix and tp, fp, tn, fn values.

To run the script in command line we need to first set the directory of test images in the format as follows:
```plaintext

data_root/

├── Non-resume/

│ │ ├── image1.jpg

│ │ ├── image2.jpg

│ │ └── ...

├── Resume/

│ │ ├── image1.jpg

│ │ ├── image2.jpg

│ │ └── ...
```
After having the data organised and the model file downloaded, we can use the test_model.py script as follows in the command line as follows:

```python
# Load the model, make predictions and obtain the classification report and confusion matrix, both visualised and saved.
train_model.py model_file_path test_folder_path target_size(default=(224, 224)) batch_size(default=16)
```
The results obtained for my test data is as shown below, with an **accuracy of 87%.**:

![image](https://github.com/ASHIPAUL27/Resume_classifier/assets/152466355/9a90b9b3-0121-4ba1-88cf-1d0067c44ac0)



