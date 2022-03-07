import os

import environ
from flask import Flask, render_template, request, flash, redirect
from werkzeug.utils import secure_filename

from predictor import ImageMaskPredictor

env = environ.Env()
environ.Env.read_env()

app = Flask(__name__)

UPLOAD_FOLDER = env("UPLOAD_FOLDER")
app.config['SECRET_KEY'] = env("SECRET_KEY")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/prediction/', methods=['POST', 'GET'])
def prediction():
    if request.method == 'GET':
        flash("Get back and send your photo.")
        return redirect("/")
    if request.method == 'POST':
        try:
            file = check_request_file(request)
            filename = secure_filename(file.filename)
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(full_filename)
            img = open(full_filename, 'rb')
            image_predictor = ImageMaskPredictor()
            prediction1 = image_predictor.classify_image(img)
            return render_template('prediction.html', pred=prediction1, user_image="/../" + full_filename)
        except NoFileException:
            flash('No file was added')
            return redirect(request.url)
        except BadFileTypeException:
            flash('Bad file type. Add png or jpg/jpeg')
            return redirect(request.url)


def check_request_file(request):
    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            if allowed_file(file.filename):
                return file
            else:
                return BadFileTypeException
        else:
            raise NoFileException
    else:
        raise NoFileException


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run()


class BadFileTypeException(Exception):
    def __init__(self):
        self.message = "Input file is not a photo"
        super().__init__(self.message)


class NoFileException(Exception):
    def __init__(self):
        self.message = "File not found exception"
        super().__init__(self.message)
