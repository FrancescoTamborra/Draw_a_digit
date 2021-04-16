# Draw a digit

"Draw a digit" is a simple Flask web app for mouse-or-finger-written digit recognition.

**Demo:[site](https://)**

<p align="center">
  <img width="1200" src="/static/images/screenshot_home.png">
</p>


## Installation
Make sure to have Python >= 3.something on your machine or virtual environment and run:

    pip install -r requirements.txt
   
## Usage

- Run the app:

      python3 app.py
    
The app will run on: [http://localhost:8080](http://localhost:8080/)

- Train the model:

      python3 model/CNN.py
    
If you want to run cross-validation and plot it you have to uncomment the lines:

    # histories = evaluate_model(trainX, trainY)
    # model_performance(histories)

## Model

The model is a 2xCNN layers with the following architecture:

![CNN architecture](/static/images/nn.png)

For more details see ``model\CNN.py``.

The test accuracy of the model is **99.3%**.
    
