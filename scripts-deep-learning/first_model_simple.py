"""
Shallow neural network model - FNN (feedforward neural networks)
"""
import pandas as pd
import numpy as np

import time

start_time = time.perf_counter()

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow import keras

from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

#############################################

filename = '../data/preproc_dataset_v4_amine.csv'

df = pd.read_csv(filename, delimiter=',')

##############################################

# Target variable
y = df["price"].values.reshape(-1, 1)  # reshape - for StandardScaler

# Remove unnecessary columns
df = df.drop(columns=["price"])

df = df.drop(columns=["postal_code"])  # Functional dependency !

# Numeric features
num_features = ["area", "build_year", "primary_energy_consumption", "cadastral_income"]
X_num = df[num_features]

# Other remaining features
other_cols = [c for c in df.columns if c not in num_features]
X_other = df[other_cols]

############################################

###  Train-Test Split ###

X_train_num, X_test_num, \
X_train_other, X_test_other, \
y_train, y_test = train_test_split(
    X_num, X_other, y, test_size=0.2
)

############################################

###  Scaling ###

num_scaler = StandardScaler()

X_train_num = num_scaler.fit_transform(X_train_num)
X_test_num = num_scaler.transform(X_test_num)

y_scaler = StandardScaler()

y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)

#############################################

### Convert other features to numpy ###
X_train_other = X_train_other.to_numpy()
X_test_other = X_test_other.to_numpy()

X_train_other = X_train_other.astype("float32")
X_test_other = X_test_other.astype("float32")

#############################################
#############################################
#############################################

#############################################
#                                           #
#           Parameters                      #
#                                           #
#############################################

EPOCHS = 35

LEARNING_RATE = 0.001

DEPTH = 3    # Number of hidden layers (2, 3)

BATCH_SIZE = 64

#############################################
#                                           #
#        Build Keras model                  #
#                                           #
#############################################

num_input = Input(shape=(X_train_num.shape[1], ), name="num_input")
other_input = Input(shape=(X_train_other.shape[1], ), name="other_input")

#############################################

input_layer = Concatenate()([num_input, other_input])

if (DEPTH == 3):

    x = Dense(128, activation="relu")(input_layer)
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    output = Dense(1)(x)

elif (DEPTH == 2):

    x = Dense(64, activation="relu")(input_layer)
    x = Dense(32, activation="relu")(x)
    output = Dense(1)(x)

else:
    raise ValueError("Invalid DEPTH value")

model = Model(inputs=[num_input, other_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse")

#############################################
#                                           #
#        Train the model                    #
#                                           #
#############################################

history = model.fit([ X_train_num, X_train_other ],
                      y_train,
                      epochs=EPOCHS,
                      batch_size=BATCH_SIZE,
                      verbose=1)

#############################################
#                                           #
#        Evaluate the model                 #
#                                           #
#############################################

y_pred_scaled = model.predict([X_test_num, X_test_other])

y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_test_original = y_scaler.inverse_transform(y_test)

r2 = r2_score(y_test_original, y_pred)
print(f"R2 on test set:{r2:.3f}")

#############################################
#############################################
#############################################

if(0):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train loss over epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

#############################################

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"\nTimer: {elapsed_time:.0f} sec\n")

#############################################

print('\nTask completed!')

#############################################
