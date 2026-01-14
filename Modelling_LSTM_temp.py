import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

#import dataset
df = pd.read_csv("student_data.csv", sep=';')

#pisahkan fitur dan target
X = df.drop('Target', axis=1)
y = df['Target']

#encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

#encode fitur kategorikal
X_encoded = pd.get_dummies(X, drop_first=True)

#scalling fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

#train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

#reshape data untuk LSTM
X_train_seq = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_seq  = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

#one-hot encoding target
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat  = to_categorical(y_test, num_classes=3)

#baseline LSTM modelling
model = Sequential()

model.add(LSTM(64, input_shape=(1, X_train_seq.shape[2])))
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

#training model
history = model.fit(
    X_train_seq,
    y_train_cat,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

#evaluasi model
loss, accuracy = model.evaluate(X_test_seq, y_test_cat)
print("Test Accuracy:", accuracy)

#classification report
from sklearn.metrics import classification_report

y_pred = model.predict(X_test_seq)
y_pred_class = np.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred_class))

#classification matrix
cm = confusion_matrix(y_test, y_pred_class)

sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
