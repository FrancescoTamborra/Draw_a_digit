import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical


# -- Utility functions

# plot performances
def model_performance(mean_histories, epochs=10, n_folds=5):

    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    # plot loss
    ax0.set_title('Cross Entropy Loss')
    ax0.plot(range(1, epochs+1), mean_histories[0], color='black')
    ax0.plot(range(1, epochs+1), mean_histories[1], color='red', linestyle='dashed')
    ax0.set_xlabel('Epochs')
    ax0.set_xticks(range(1, epochs+1))
    ax0.legend(['train', 'validation'], loc='upper right')
    # plot accuracy
    ax1.set_title('Accuracy')
    ax1.plot(range(1, epochs+1), mean_histories[2], color='black')
    ax1.plot(range(1, epochs+1), mean_histories[3], color='red', linestyle='dashed')
    ax1.set_xlabel('Epochs')
    ax1.set_xticks(range(1, epochs+1))
    ax1.legend(['train', 'validation'], loc='lower right')

    fig.suptitle('{}-fold cross-validation'.format(n_folds))
    fig.savefig('perf.png')


# score
def score(model, testX, testY):
    sco = model.evaluate(
        testX,
        to_categorical(testY)
    )
    result = 'This model achieved {:.3f} test loss and {:.2f} % test accuracy'.format(
        sco[0], sco[1]*100)
    return result


# k-fold cross-validation
def evaluate_model(model, dataX, dataY, n_folds=5, epochs=10, verbose=0, batch_size=32):
    print('evaluating the model...')
    histories = list()
    kfold = KFold(n_folds, shuffle=True, random_state=42)
    for train_index, test_index in kfold.split(dataX):
        print('new fold')
        trainX, testX = dataX[train_index], dataX[test_index]
        trainY, testY = dataY[train_index], dataY[test_index]
        # fit model
        history = model.fit(
            trainX,
            to_categorical(trainY),
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(testX, to_categorical(testY)),
            verbose=verbose
        )
        histories.append(history)
    return histories


# save k-folds histories
def save_histories(histories):
    list_histories = []
    for h in histories:
        # we have to json.dumps because a list of histories is not serializable :(
        list_histories.append(json.dumps(h.history))

    json.dump(list_histories, open('list_histories.txt', 'w'), indent=4)


# calculate mean history over folds
def mean_folds_history(n_folds=5, epochs=10):
    # load histories from file:
    with open('list_histories.txt') as json_file:
        data = json.load(json_file)

    # Let's build a tensor with size: (nb_attr, nb_folds, nb_epochs)
    T = np.zeros(shape=(4, n_folds, epochs))

    for idx, h in enumerate(data):
        hj = json.loads(h.replace("''", '"'))  # because of json.dumps
        T[0][idx] = hj['loss']
        T[1][idx] = hj['val_loss']
        T[2][idx] = hj['accuracy']
        T[3][idx] = hj['val_accuracy']

    # average over folds
    histories_av = np.mean(T, axis=1)

    return histories_av

# --


# Dataset loading and pre-treatment
print('loading dataset...')
(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = (trainX / 255) - 0.5
testX = (testX / 255) - 0.5
trainX = np.expand_dims(trainX, axis=3)
testX = np.expand_dims(testX, axis=3)


# define CNN model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=5, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D())
    model.add(Dropout(0.4))
    model.add(Conv2D(64, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


print('defining the model...')
model = define_model()
print('model created:')
print(model.summary())

# save the model summary
with open('model_summary.txt', 'w') as fh:
    model.summary(print_fn=lambda x: fh.write(x + '\n'))


# Uncomment the 4 lines below to perform cross-validation, save the histories and produce a plot on the performance

# histories = evaluate_model(model, trainX, trainY, verbose=2)
# save_histories(histories)
# histories_mean = mean_folds_history()
# model_performance(histories_mean)


# Test the model
print('fitting the model...')
model.fit(
    trainX,
    to_categorical(trainY),
    epochs=10,
    batch_size=32
)

# score
print('scoring the model...')
result = score(model, testX, testY)
print(result)

# save model
model.save('32C5-P2_64C5-P2-128.h5')
