import numpy as np

def aug_aws(train, train_labels, n_draws, max_slice_ratio=0.4, min_slice_length=3):
    """
    Augmentation method: AWS - Activity Window Slicing
    
    Authors:
    Dawid Warchoł, Mariusz Oszust
    
    This function augments multivariate time-series data by intelligent slicing.
    Instead of removing a random excerpt (window), it identifies and removes the excerpt
    with the least amount of activity or change (potentially the least significant one).
    
    The process is as follows:
    1. For each time series, an "activity score" is calculated for each time step.
       This score is the sum of absolute differences of all features between
       consecutive time steps.
    2. An excerpt of a random length is determined based on the provided parameters.
    3. The function finds the position of this excerpt where the cumulative activity
       score is minimized.
    4. This low-activity subsequence is removed to create an augmented sample.
    5. Therefore, the method focuses on the most dynamic and informative
       parts of the time series.
    
    Parameters:
    - train: list of 2D numpy arrays. Expected shape per sample: (features, time).
             If (time, 1) is detected, it will be transposed automatically.
    - train_labels: list or array of corresponding class labels.
    - n_draws: number of augmented versions to generate per sample.
    - max_slice_ratio: max length of slice as fraction of total length.
    - min_slice_length: min length of slice in time steps.
    
    Returns:
    - out_train: numpy array of augmented samples (dtype=object).
                 Samples will be shorter than input samples (slicing).
    - out_train_labels: numpy array of corresponding labels.
    """
    
    out_train_list = []
    out_train_labels_list = []
    
    # Iterate over the training data
    # Handles input as list of arrays, object array, or dense 3D array
    for i in range(len(train)):
        temp = train[i]
        
        # Ensure temp is a numpy array (handles cases where input might be a list of lists)
        temp = np.array(temp)
        
        # Dimensions: (features, time)
        n_samples = temp.shape[1] 
        current_label = train_labels[i]

        # Augmentation is not performed if the sequence is too short
        if n_samples < 4 or n_samples <= min_slice_length:
            for _ in range(n_draws):
                out_train_list.append(temp.copy())
                out_train_labels_list.append(current_label)
            continue

        # Stage 1: Calculate activity scores for the entire sequence.
        # Sum of absolute differences between consecutive time steps.
        activity_scores = np.sum(np.abs(np.diff(temp, axis=1)), axis=0)

        for _ in range(n_draws):
            # Stage 2: Define the size of the window to remove based on parameters.
            max_excerpt_length = int(np.floor(n_samples * max_slice_ratio))
            
            # Ensure that the minimum length is not greater than the maximum length
            if min_slice_length > max_excerpt_length:
                max_excerpt_length = min_slice_length
            
            # Ensure a window larger than the sequence itself is not removed
            if max_excerpt_length >= n_samples:
                max_excerpt_length = n_samples - 1
            
            # Augmentation is not performed if slicing is not possible with given parameters
            if max_excerpt_length < min_slice_length:
                out_train_list.append(temp.copy())
                out_train_labels_list.append(current_label)
                continue
            
            # Random integer between min (inclusive) and max (inclusive)
            excerpt_length = np.random.randint(min_slice_length, max_excerpt_length + 1)
            cut_a = 0
            
            # Perform activity analysis only if it's meaningful
            if 1 < excerpt_length < n_samples:
                min_activity_sum = float('inf')
                best_cut_start = 0
                
                # Stage 3: Use a sliding window to find the excerpt with minimum activity.
                num_possible_starts = n_samples - excerpt_length + 1
                
                for s in range(num_possible_starts):
                    start_idx = s
                    end_idx = s + excerpt_length - 1
                    
                    if start_idx >= end_idx: 
                        current_activity_sum = 0
                    else:
                        current_activity_sum = np.sum(activity_scores[start_idx:end_idx])

                    if current_activity_sum < min_activity_sum:
                        min_activity_sum = current_activity_sum
                        best_cut_start = s
                
                cut_a = best_cut_start
            else:
                # Slice randomly if excerpt length is 0 or takes up whole sample
                if n_samples > excerpt_length:
                    cut_a = np.random.randint(0, n_samples - excerpt_length + 1)
            
            cut_b = cut_a + excerpt_length
            
            # Stage 4: Remove the selected low-activity excerpt
            part1 = temp[:, :cut_a]
            part2 = temp[:, cut_b:]
            temp_augmented = np.hstack((part1, part2))

            if temp_augmented.size > 0:
                out_train_list.append(temp_augmented)
            else:
                out_train_list.append(temp.copy())
            
            out_train_labels_list.append(current_label)

    # Convert list to NumPy object array.
    # We create an empty object array and fill it manually.
    # This avoids ValueError during broadcasting if arrays have similar shapes but are not identical.
    out_train = np.empty(len(out_train_list), dtype=object)
    out_train[:] = out_train_list

    out_train_labels = np.array(out_train_labels_list)

    return out_train, out_train_labels