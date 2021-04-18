# Draw a digit

"Draw a digit" is a simple Flask web app for mouse-or-finger-written digit recognition.

<br>

**Demo:[https://draw-a-digit.nw.r.appspot.com/](https://draw-a-digit.nw.r.appspot.com/)**


<p align="center">
  <img width="1200" src="/static/images/demo_gif.gif">
</p>


## Installation
Make sure you have Python3.7+ on your machine or virtual environment.

It is recommended to upgrade pip:

    pip install --upgrade pip

then just run:

    pip install -r requirements.txt

## Usage

Run the app:

    python3 app.py

The app will run on: [http://localhost:8080](http://localhost:8080/)

## Retrain the model

The model is already trained but if you want to modify something and retrain the network you first need to: 

- add these two lines to ``requirements.txt`` :

      matplotlib
      scikit-learn

- execute again:

      pip install -r requirements.txt


- launch the training with:

      python3 model/CNN.py

If you want to run cross-validation, save it and make a plot of the average performance before the training, you have to uncomment these lines:

    # histories = evaluate_model(model, trainX, trainY)
    # save_histories(histories)
    # histories_mean = mean_folds_history()
    # model_performance(histories_mean)

take care of updating and pass to the functions the new default parameters if you changed them (e.g. n_folds, epochs)

## Model

The model is a 2xCNN layers with the following architecture:

![CNN architecture](/static/images/nn.png)

For more details see ``model\CNN.py``.

The 5-folds cross-validation accuracy of the model reached **99.5%** after 10 epochs.
