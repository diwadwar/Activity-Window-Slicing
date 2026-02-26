# Activity Window Slicing
Activity Window Slicing (AWS) is a method for multivariate time series datasets augmentation.

# How does it work?
Unlike standard approaches based on random cropping, AWS analytically identifies and removes signal segments characterized by the lowest dynamic activity, ensuring the model focuses on the most informative sequences.

# Languages
AWS is available in Python and Matlab: aug_aws.py and aug_aws.mat. Scripts with an example of use (example.py and example.m) are also provided.

# Polish Sign Language (PSL) dataset
PSL is a sample dataset created to demonstrate hwo to use AWS and how effective it is.

PSL, consists of six Polish Sign Language gestures: 'good morning', 'goodbye', 'greetings', 'please', 'thank you', and 'why?'. Gestures were performed by four people seated in a fixed position relative to an RGB camera. Each person performed each gesture five times. Half of the dataset (people with numbers 1 and 3) are used as the training subset by default, while the other half (people with numbers 2 and 4) are used as the testing subset. Then, the Pose Detection module of the MediaPipe library was used to generate 3D coordinates of body landmarks (characteristic body parts). Only the torso and arm landmarks were used, while the head and legs were skipped due to the specificity of sign language.

PSL is available in text format ("PSL dataset" directory), in Python numpy format ("AWS - Python/PSL.npz"), and in Matlab format ("AWS - Matlab/PSL.mat").
