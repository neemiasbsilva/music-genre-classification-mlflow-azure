import json
import numpy as np
from sklearn.model_selection import train_test_split
import librosa
import requests
import wget

def get_spectrogram(data):
    inputs = []

    for x in data:
        delta = librosa.feature.delta(x)
        delta2 = librosa.feature.delta(delta)
        s = np.dstack(
        (x, delta, delta2))
        inputs.append(s)

    return np.asarray(inputs)


def get_dataset(path_dataset):

    filename = wget.download(path_dataset)

    with open(filename, 'r') as fp:
        dataset = json.load(fp)

    inputs = get_spectrogram(np.asarray(dataset["mfcc"]))
    targets = np.array(dataset["labels"])

    return inputs, targets

def preprocessing(path_dataset, test_size=0.25, val_size=0.2):

    data, labels = get_dataset(path_dataset)

    X_train, X_test, y_train, y_test = train_test_split(data, labels,
                                            test_size=test_size,
                                            stratify=labels,
                                            random_state=777,
                                            shuffle=True)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                        test_size=val_size,
                                                        random_state=777)
    
    X_train = X_train[..., np.newaxis] 
    X_val = X_val[..., np.newaxis]  
    X_test = X_test[..., np.newaxis]

    return X_train, X_val, X_test, y_train, y_val, y_test
