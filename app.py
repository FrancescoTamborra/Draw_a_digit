from flask import Flask, render_template, request, url_for, redirect

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def root():
    return render_template(
        'index.html'
    )



if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
