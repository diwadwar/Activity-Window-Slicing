%% Augmentation method: AWS - Activity Window Slicing
%
% Authors:
% Dawid Warchoł, Mariusz Oszust
% 
% This function augments multivariate time-series data by intelligent slicing.
% Instead of removing a random excerpt (window), it identifies and removes the excerpt
% with the least amount of activity or change (potentially the least significant one).
%
% The process is as follows:
% 1. For each time series, an "activity score" is calculated for each time step.
%    This score is the sum of absolute differences of all features between
%    consecutive time steps.
% 2. An excerpt of a random length is determined based on the provided parameters.
% 3. The function finds the position of this excerpt where the cumulative activity
%    score is minimized.
% 4. This low-activity subsequence is removed to create an augmented sample.
% 5. Therefore, the method focuses on the most dynamic and informative
%    parts of the time series.
%
% Inputs:
% - train: cell array of multivariate time-series samples (dim × time)
% - trainLabels: corresponding class labels (numerical 1D array)
% - nDraws: number of augmented versions to generate per sample
% - maxSliceRatio: the maximum length of the slice as a fraction
%                  of the total sequence length; 
%				   default: 0.4
% - minSliceLength: the minimum length of the slice in time steps;
%                   default: 3
%
% Outputs:
% - outTrain: augmented training data
% - outTrainLabels: corresponding labels for the augmented data

function [outTrain, outTrainLabels] = aug_aws(train, trainLabels, nDraws, maxSliceRatio, minSliceLength)
    % Default parameter values
    if nargin < 4
        maxSliceRatio = 0.4;
    end
    if nargin < 5
        minSliceLength = 3;
    end

    % Output cell arrays 
    outTrain = cell(1, length(trainLabels) * nDraws);
    outTrainLabels = zeros(1, length(trainLabels) * nDraws);
    xSize = length(train);

    counter = 1;
    for i = 1:xSize
        temp = train{i};
        nSamples = size(temp, 2);

        % Augmentation is not performed if the sequence is too short
        if nSamples < 4 || nSamples <= minSliceLength
            for iD = 1:nDraws
                outTrain{counter} = temp;
                outTrainLabels(counter) = trainLabels(i);
                counter = counter + 1;
            end
            continue;
        end
        
        % Stage 1: Calculate activity scores for the entire sequence.
        activity_scores = sum(abs(diff(temp, 1, 2)), 1);

        for iD = 1:nDraws
            % Stage 2: Define the size of the window to remove based on parameters.
            maxExcerptLength = floor(nSamples * maxSliceRatio);
            
            % Ensure that the minimum length is not greater than the maximum length
            if minSliceLength > maxExcerptLength
                maxExcerptLength = minSliceLength;
            end
            
            % Ensure a window larger than the sequence itself is not removed
            if maxExcerptLength >= nSamples
                maxExcerptLength = nSamples - 1;
            end
            
            % Augmentation is not performed if slicing is not possible with the given parameters
            if maxExcerptLength < minSliceLength
                outTrain{counter} = temp;
                outTrainLabels(counter) = trainLabels(i);
                counter = counter + 1;
                continue;
            end
            
            excerptLength = randi([minSliceLength, maxExcerptLength]);

			% Default start index
            cutA = 1; 
            
            % Perform activity analysis only if it's meaningful
            if excerptLength < nSamples && excerptLength > 1
                min_activity_sum = inf;
                best_cut_start = 1;
                
                % Stage 3: Use a sliding window to find the excerpt with the minimum activity.
                num_possible_starts = nSamples - excerptLength + 1;
                for s = 1:num_possible_starts
                    % The indices in activity_scores correspond to changes between samples.
                    % A window from 's' to 's+excerptLength-1' in 'temp' corresponds
                    % to activity scores from 's' to 's+excerptLength-2'.
                    activity_indices = s : (s + excerptLength - 2);
                    
                    % It's possible for a slice at the very end to have no activity indices
                    % (e.g., excerptLength=1). Handle this case.
                    if isempty(activity_indices)
                        current_activity_sum = 0;
                    else
                        current_activity_sum = sum(activity_scores(activity_indices));
                    end

                    if current_activity_sum < min_activity_sum
                        min_activity_sum = current_activity_sum;
                        best_cut_start = s;
                    end
                end
                cutA = best_cut_start;
            else
                % Slice randomly if the excerpt length is equal to 0 or nSamples.
                if nSamples > excerptLength
                    cutA = randi([1, nSamples - excerptLength + 1]);
                end
            end
            
            cutB = cutA + excerptLength - 1;

            % The excerpt to be removed
            excerpt = cutA:cutB;

            % Stage 4: Remove the selected low-activity excerpt
            tempAugmented = temp;
            tempAugmented(:, excerpt) = [];

            % Store the augmented sample and its corresponding label
            if ~isempty(tempAugmented)
                outTrain{counter} = tempAugmented;
            else
                outTrain{counter} = train{i};
            end
            outTrainLabels(counter) = trainLabels(i);
            counter = counter + 1;
        end
    end
    outTrain = outTrain';
    outTrainLabels = outTrainLabels';
end