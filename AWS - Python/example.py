import numpy as np
from aug_aws import aug_aws

dataset = np.load('PSL.npz', allow_pickle=True)

data = dataset['data'] 
labels = dataset['labels']

# first cell and third cell belong to the training subset
train = np.concatenate((data[0], data[2]))
trainLabels = np.concatenate((labels[0], labels[2]))

multiplicity = 1

# augmentation
outTrain, outTrainLabels = aug_aws(train, trainLabels, multiplicity)