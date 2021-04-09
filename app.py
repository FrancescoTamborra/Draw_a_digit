from flask import Flask, render_template, request, url_for, redirect
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

app = Flask(__name__)
model = None


def model_loader():
    global model
    model = load_model('model/final_model.h5')
    print(model.summary())


@app.route('/', methods=['GET', 'POST'])
def root():
    return render_template(
        'index.html'
    )


# model_loader()

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
