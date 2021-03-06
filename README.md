# Satellite Pose estimation using Convolutional Neural Networks
# Group 23 - ECE 228

Pose estimation of known uncooperative spacecraft plays an important role in various satellite servicing missions of scientific, economic, and societal benefits. In this work, we aim to estimate the pose, i.e., the relative position and altitude, of a known spacecraft from individual grayscale images using deep neural networks. Convolutional neural networks have been widely exploited for image processing tasks such as object detection, classification, human pose estimation and action recognition. 

Dataset Competition Website - https://kelvins.esa.int/satellite-pose-estimation-challenge/home/

# I) Downloading the dataset and extracting to folder
- Download the data from : https://kelvins.esa.int/satellite-pose-estimation-challenge/data/ 
![alt text](https://raw.githubusercontent.com/arun1993/satellite_pose_estimation_ece228/master/Satellite_image_dataset.png)
The downloaded zip file contains dataset in images/ folder. 
      train/: a folder containing 12000 synthetic images for training. Images are 8 bit monochrome in jpeg format, with a resolution of 1920×1200 pixels.
      test/: 2998 similar synthetic images for evaluating submissions
      real_test/: 300 real images of the Tango satellite mock-up, same format and resolution as the synthetic images.
      real/: 5 example real images, with pose labels
      train.json: filenames and corresponding pose labels for the 12000 training images
      test.json: list of filenames of the test images
      real_test.json: list of filenames of the real test images
      real.json: filenames and corresponding pose labels for five example real image
- Extract to current folder where scripts are : as ./speed folder.

# II) Running top different models - training scripts 

Libraries installation:
======================
pip install numpy pillow matplotlib
pip install torch torchvision  
pip install tensorflow-gpu  
pip install jupyter  

Training:
=========
python [path to python file] --dataset [path to downloaded dataset] --epochs [num epochs] --batch [batch size]

As the training is finished, the model is evaluated on all images of the training and real_training sets, and a CSV file is generated that can be directly submitted on the competition page.
- a) Run poseloss_new_arch.py with default command line arguments. Specify path to where data is otherwise.
- b) Run linear_weighted_loss model : <add>

# III) Generating .csv file results and uploading to Kaggle 

# Leaderboard standing:
![alt text](https://raw.githubusercontent.com/arun1993/satellite_pose_estimation_ece228/master/kaggle_leaderboard.JPG)
