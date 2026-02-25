load('PSL.mat'); 
trainingData = [data{1}; data{3}]; % first cell and third cell belong to the training subset

% converting data to appropriate format
train = cellfun(@(x) x', trainingData(:,1), 'UniformOutput', false);
trainLabels = str2double(trainingData(:,2));

multiplicity = 1;

% augmentation
[outTrain, outTrainLabels] = aug_aws(train, trainLabels, multiplicity);