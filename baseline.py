import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib

def softmax(x):
    # Compute the exponential values for each element in the input array
    exps = np.exp(x - np.max(x))

    # Compute the softmax values by dividing the exponential of each element by the sum of exponentials
    return exps / np.sum(exps)


# Load the data from a Parquet file into a pandas DataFrame.
data_frame = pd.read_parquet(sys.argv[1])


X = np.array([[data_frame['raw_prediction'].values[i]] for i in range(data_frame.shape[0])]).squeeze()
model = joblib.load('model.pkl')
data_frame['prediction'] = model.predict(X)
y_prob = model.predict_proba(X).tolist()
y_prob = list(map(list, zip(*y_prob)))
data_frame['pred_probability'] = y_prob[0]

# Initialize an empty list to store the maximum confidence values.
max_confidences = []

# Iterate over the DataFrame rows.
for _, row in data_frame.iterrows():
    # Compute softmax for the 'raw_prediction' column of the current row.
    softmax_values = softmax(row['raw_prediction'])
    
    # Find the maximum confidence value and append it to the list.
    max_confidences.append(softmax_values.max())

# Add a new column 'confidence' to the DataFrame using the list of maximum confidence values.
data_frame['confidence'] = max_confidences
data_frame['pred'] = [x.argmax() for x in data_frame['raw_prediction']]

# Sort the DataFrame.
sorted_data_frame = data_frame.loc[data_frame['prediction']==1]
sorted_data_frame = sorted_data_frame.loc[sorted_data_frame['pred_probability']==1]
sorted_data_frame = sorted_data_frame.sort_values(by='confidence', ascending=False)

# Determine the number of top records to consider for computing mean distance.
top_records_count = int(0.1 * len(data_frame))

pd.Series(sorted_data_frame.iloc[:top_records_count].index).to_csv('submission.csv', index=False, header=None)

print(f'Submission saved to submission.csv, number of rows: {top_records_count}')
