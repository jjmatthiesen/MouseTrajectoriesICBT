# The Promise and Challenges of Computer Mouse Trajectories in DMHIs – A Feasibility Study on Pre-Treatment Dropout Predictions 

*Authors will be added after blind review.*

## Mouse Trajectories in Digital Mental Health Interventions
This repository provides the code to the paper.
We introduce the novel data type of mouse trajectories within the context of Digital Mental Health Interventions (DMHIs).
The paper discusses how to gather and process mouse trajectory data on questionnaires in DMHIs.


## File overview

### 🗀 models
#### 🗎 [cnn1d.py](./models/cnn1d.py)
Contains the architecture for a 1D Convolutional Neural Network (1D-CNN).

#### 🗎 [cnn1d_antal_feher.py](./models/cnn1d_antal_feher.py)
Implements the architecture for a 1D-CNN, as used in [Antal et al.(2020)](https://ieeexplore.ieee.org/document/9465583).

#### 🗎 [cnn1d_encode.py](./models/cnn1d_encode.py)
Implements a 1D-CNN architecture similar to the encoder part of an autoencoder.

#### 🗎 [nonseq_models.py](./models/nonseq_models.py)
Utility class for non-sequential scikit-learn models.

### 🗀 src
#### 🗎 [aggregated_learning.py](src/aggregated_learning.py)
Runs the deep learning experiment using different hyperparameter setting and models.

#### 🗎 [train_nonseq_models.py](src/train_nonseq_models.py)
Runs the non-sequential experiment using different hyperparameter settings for baseline, 10 and 3 mouse features.

### 🗀 scr/pre_processing

#### 🗎 [preprocess_outcomes.py](./src/pre_processing/preprocess_outcomes.py)
Very specific to this project as highly dependent on the data structure from the studies system. Returns DataFrame with
- Only the patients that filled out the baseline questionnaire within a specific time frame
- Includes dropout_mod = outcome variable
- Includes all baseline features

#### 🗎 [mouse_features.py](./src/pre_processing/mouse_features.py)
Takes the raw mouse trajectory from the [tracker](https://github.com/jjmatthiesen/evtrack/tree/setup_karolinskaInstitutet) and creates features such as:
- jitter, jitter2 (see appendix for mathematical definition)
- x_min_max_diff, y_min_max_diff
- percentage of screen usage (total, x and y)
- angle_min, angle_mean, angle_max, angle_change_max, angle_change_mean, angle_speed_max,angle_speed_min, angle_speed_mean, 
- angle_change_speed_max, angle_change_speed_min, angle_change_speed_mean
- curvature_avg, curvature_change_mean,
- acute_angles, obtuse_angles, 
- speed_max, speed_mean, horizontal_speed_min, horizontal_speed_max, horizontal_speed_avg, vertical_speed_min, vertical_speed_max, vertical_speed_avg, 
- move_time_total, 
- movement_duration_ratio, 
- pause_time_total,
- moved_dist, moved_dist_norm,
- number_dp (number of data points)
- pauses_no,
- clicks_no,
- scroll_speed_min,scroll_speed_max, scroll_speed_mean
- dispersal_x_percent, dispersal_y_percent, area_percent

#### 🗎 [participant_ids.py](./src/pre_processing/participant_ids.py)
Provides an overview of participant numbers per treatment, time point, and mobile/desktop usage.

#### 🗎 [pre_process_files.py](./src/pre_processing/pre_process_files.py)
Pre-processes of the raw mouse data 
    - Sorting files into desktop and mobile user categories
    - Within those folders, separating data into train_test (users with labels) and pre_train (users without labels)
    - Within train_test, further separating data into dropout or non_dropout categories according to the labels
    - Further categorizing dropout/non_dropout data into Pre and Screening based on the time of recording
    - Creating a folder for each user within these categories to store their session files

#### 🗎 [select_features.py](./src/pre_processing/select_features.py)
Selects certain features from the whole feature list created with [mouse_features.py](./src/pre_processing/mouse_features.py).
    - speed_avg, 
    - angle_change_mean,
    - acute_angles,
    - obtuse_angles,
    - jitter,
    - pause_time_total,
    - moved_dist,
    - number_dp, 
    - pauses_no, 
    - scroll_speed_mean

#### 🗎 [pre_process_nonseq.py](./src/pre_processing/pre_process_nonseq.py)
Pulls non-sequential features incl. imputing missing MADRS-S values

### 🗀 scr/utils
#### 🗎 [data.py](./src/utils/data.py)
Includes a class for creating datasets and functions for generating artificial trajectories using Gaussian Processes (GPs).

#### 🗎 [globals.py](./src/utils/globals.py)
Contains global variables, including:
- A dictionary mapping all assessments of the intervention
- A dictionary mapping mouse events
- A map categorizing interventions

#### 🗎 [imports.py](./src/utils/imports.py)
Lists all packages used in this project.

#### 🗎 [imports_nn.py](./src/utils/imports_nn.py)
Lists additional packages for deep learning. 
Note that some of these packages are large, so if you do not plan to run deep learning experiments, you do not need to install them.

#### 🗎 [plots.py](./src/utils/plots.py)
Includes all plotting functions used in this project.

#### 🗎 [train_test.py](./src/utils/train_test.py)
Contains functions for training and testing of the neural networks.

#### 🗎 [utils.py](./src/utils/utils.py)
Provides smaller utility functions, such as a logger, metric collector, and Euclidean distance calculation. 

---
## Setup
Install the packages from requirements.txt

> pip install -r requirements.txt

##  Run
To run the code, follow these steps:

1. Record data using [this tracker](https://github.com/jjmatthiesen/evtrack/tree/setup_karolinskaInstitutet).
2. Create a folder data/mouse_data/raw/ and move the mouse data and metadata there.
3. Place your labels file in /data/outcomes.csv.
4. Run [pre_process_files.py](src/pre_processing/pre_process_files.py).
5. Generate mouse features with [mouse_features.py](src/pre_processing/mouse_features.py).
6. Select features with [select_features.py](src/pre_processing/select_features.py).
7. run experiments with [aggregated_learning.py](src/aggregated_learning.py)

