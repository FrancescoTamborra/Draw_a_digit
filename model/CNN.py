import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

# load dataset and reshape (pixel value in the interval [-0.5, 0.5])
print('loading dataset...')
(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = (trainX / 255) - 0.5
testX = (testX / 255) - 0.5
trainX = np.expand_dims(trainX, axis=3)
testX = np.expand_dims(testX, axis=3)


# define CNN model
print('defining the model...')


def define_model():
    num_filters = 8
    filter_size = 3
    pool_size = 2
    model = Sequential([
        Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=pool_size),
        Dropout(0.5),
        Flatten(),
        Dense(10, activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


print('model created:')
print(define_model().summary())

# save the model summary
with open('model_summary.txt', 'w') as fh:
    define_model().summary(print_fn=lambda x: fh.write(x + '\n'))

# evaluate model with k-fold cross-validation
print('evaluating the model...')


def evaluate_model(dataX, dataY, n_folds=5):
    histories = list()
    kfold = KFold(n_folds, shuffle=True, random_state=42)
    for train_index, test_index in kfold.split(dataX):
        model = define_model()
        trainX, testX = dataX[train_index], dataX[test_index]
        trainY, testY = dataY[train_index], dataY[test_index]
        # fit model
        history = model.fit(
            trainX,
            to_categorical(trainY),
            epochs=5,
            batch_size=32,
            validation_data=(testX, to_categorical(testY))
        )
        histories.append(history)
    return histories


histories = evaluate_model(trainX, trainY)


# plot performances
def model_performance(histories):
    fig = plt.figure(figsize=(15, 5))
    for i in range(len(histories)):
        # plot loss
        plt.subplot(1, 2, 1)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='blue')
        plt.plot(histories[i].history['val_loss'], color='orange')
        plt.xticks([0, 1, 2, 3, 4])
        plt.legend(['train', 'test'], loc='upper right')
        # plot accuracy
        plt.subplot(1, 2, 2)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue')
        plt.plot(histories[i].history['val_accuracy'], color='orange')
        plt.xticks([0, 1, 2, 3, 4])
        plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    fig.savefig('model_performance_kfold_crossval.png')


model_performance(histories)


final_model = define_model()
# fit model
final_model.fit(
    trainX,
    to_categorical(trainY),
    epochs=10,
    batch_size=32
)
# score
score = final_model.evaluate(
    testX,
    to_categorical(testY)
)
print('This model achieved {:.3f} test loss and {:.2f} % accuracy'.format(score[0], score[1]*100))
# save model
final_model.save('final_model.h5')


# # load model
# 	model = load_model('final_model.h5')

# # load and prepare the image
# def load_image(filename):
# 	# load the image
# 	img = load_img(filename, grayscale=True, target_size=(28, 28))
# 	# convert to array
# 	img = img_to_array(img)
# 	# reshape into a single sample with 1 channel
# 	img = img.reshape(1, 28, 28, 1)
# 	# prepare pixel data
# 	img = img.astype('float32')
# 	img = img / 255.0
# 	return img
