import pandas as pd
import numpy as np
import sys
import joblib


def softmax(x):
    # Compute the exponential values for each element in the input array
    exps = np.exp(x - np.max(x))

    # Compute the softmax values by dividing the exponential of each element by the sum of exponentials
    return exps / np.sum(exps)


# Load the data from a Parquet file into a pandas DataFrame, approximate distances for every pair of labels into 'dist' and approximate labels' coordinates into 'coord'.
data_frame = pd.read_parquet(sys.argv[1])
dist = np.load('distances.npy')       #Unknown distances are defined as 30000
coord = np.load('coordinates.npy')

# Initialize an empty list to store the maximum confidence values and an empty array for the model's input data.
max_confidences = []
X = np.ones((len(data), 3015), float)

data_frame['predicted_dist'] = data_frame['distance']
i = 0
# Iterate over the DataFrame rows.
for _, row in data_frame.iterrows():
    
    pred = np.argmax(row['raw_prediction'])
    X_sorted_labels = np.argsort(row['raw_prediction'])
    X_distances = []
    for j in range(3000):
        if dist[pred][X_sorted_labels[j]] != 30000:
            X_distances.append(dist[pred][X_sorted_labels[j]])
            
    # Mark the items where distances between 'pred' and the rest of the labels are not defined            
    if len(X_distances)==0:
        X[i][0] = 30000
        data_frame.iat[i, 6] = 30000    #the 6th column is 'predicted_dist'
        
    else:
        X_dist_len10 = X_distances[2008:]
        X_dist_len10.sort()
        X[i][:9] = X_dist_len10[1:]
        
    X[i][9] = coord[pred][0]
    X[i][10] = coord[pred][1]
    X[i][11] = row['confidence']
    X[i][12] = row['text'].count('https://')
    X[i][13] = row['text'].count(' ')
    X[i][14] = len(row['text'])
    X[i][15:3015] = row['raw_prediction']
    i +=1
    
    # Compute softmax for the 'raw_prediction' column of the current row.
    softmax_values = softmax(row['raw_prediction'])
    
    # Find the maximum confidence value and append it to the list.
    max_confidences.append(softmax_values.max())

# Add a new column 'confidence' to the DataFrame using the list of maximum confidence values.
data_frame['confidence'] = max_confidences
data_frame['pred'] = [x.argmax() for x in data_frame['raw_prediction']]

# Sort out the marked items
X = X[X[:, 0] != 30000]
data = data_frame.loc[data_frame['predicted_dist']<30000]
data_ = data_frame.loc[data_frame['predicted_dist']==30000]

model = joblib.load('model.pkl')
data['predicted_dist'] = model.predict(X)

data_frame = pd.concat([data, data_], sort=False, axis=0)

# Sort the DataFrame.
sorted_data_frame = data_frame.sort_values(by='predicted_dist', ascending=True)

# Determine the number of top records to consider for computing mean distance.
top_records_count = int(0.1 * len(data_frame))

pd.Series(sorted_data_frame.iloc[:top_records_count].index).to_csv('submission.csv', index=False, header=None)

print(f'Submission saved to submission.csv, number of rows: {top_records_count}')
