import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the data from CSV
df = pd.read_csv('iris2.csv')

# Split features and labels
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species'].values.reshape(-1, 1)

# One-hot encode the species labels
encoder = OneHotEncoder(sparse=False)
y_encoded = encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Normalize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the feed-forward neural network
model = Sequential([
    Dense(10, input_shape=(X_train.shape[1],), activation='relu'),  # First hidden layer
    Dense(8, activation='relu'),  # Second hidden layer
    Dense(3, activation='softmax')  # Output layer (3 classes for the species)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=5, validation_split=0.1)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy:.2f}')






# # Create a virtual environment
# python -m venv venv

# # Activate the virtual environment
# venv\Scripts\activate  # On Windows

# # Install TensorFlow
# pip install tensorflow
