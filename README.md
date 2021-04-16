# Draw a digit

"Draw a digit" is a simple Flask web app for mouse-or-finger-written digit recognition.

**Demo:[site](https://)**

## Installation
Make sure to have Python >= 3.something on your machine or virtual environment and run:

    pip install -r requirements.txt
    
## Usage

From terminal:

    python3 app.py
    
The app will run on: [http://localhost:8080](http://localhost:8080/)

## Model

The model is a 2xCNN layers with the following architecture:

![CNN architecture](/static/images/nn.png)

See ``model\CNN.py`` for more details.
    
