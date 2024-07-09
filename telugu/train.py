import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from data.pickle
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = data_dict['data']
labels = data_dict['labels']

# Define a fixed number of landmarks (adjust as needed)
max_landmarks = 42

# Pad or truncate each data point to have max_landmarks landmarks
for i in range(len(data)):
    data[i] = data[i][:max_landmarks] + [[0, 0]] * (max_landmarks - len(data[i]))

# Convert to NumPy array
data = np.asarray(data)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Create and train the RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate the model
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the model to model.p
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)



