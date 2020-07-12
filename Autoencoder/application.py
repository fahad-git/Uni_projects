import os

from flask import Flask, request, render_template, send_from_directory, Response
from skimage.transform import resize, rescale
import matplotlib.pyplot as plt
import numpy as np
import shutil
import time

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import Conv2DTranspose, UpSampling2D, add
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import tensorflow as tf

print(tf.__version__)

__author__ = 'fahad'

# app = Flask(__name__, static_folder="images")
application = app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

encoder = None
auto_encoder = None

def get_model():
    global encoder, auto_encoder
    print(" * Loading Keras model...")

    input_img = Input(shape=(256, 256, 3))
    l1 = Conv2D(64, (3, 3), padding='same', activation='relu',
                activity_regularizer=regularizers.l1(10e-10))(input_img)

    l2 = Conv2D(64, (3, 3), padding='same', activation='relu',
                activity_regularizer=regularizers.l1(10e-10))(l1)

    # By default max pooling is 2 by 2
    l3 = MaxPooling2D(padding='same')(l2)

    l4 = Conv2D(128, (3, 3), padding='same', activation='relu',
                activity_regularizer=regularizers.l1(10e-10))(l3)
    l5 = Conv2D(128, (3, 3), padding='same', activation='relu',
                activity_regularizer=regularizers.l1(10e-10))(l4)
    l6 = MaxPooling2D(padding='same')(l5)
    l7 = Conv2D(256, (3, 3), padding='same', activation='relu',
                activity_regularizer=regularizers.l1(10e-10))(l6)

    encoder = Model(input_img, l7)

    print(encoder.summary())

    input_img = Input(shape=(256, 256, 3))
    l1 = Conv2D(64, (3, 3), padding='same', activation='relu',
                activity_regularizer=regularizers.l1(10e-10))(input_img)

    l2 = Conv2D(64, (3, 3), padding='same', activation='relu',
                activity_regularizer=regularizers.l1(10e-10))(l1)

    # By default max pooling is 2 by 2
    l3 = MaxPooling2D(padding='same')(l2)

    l4 = Conv2D(128, (3, 3), padding='same', activation='relu',
                activity_regularizer=regularizers.l1(10e-10))(l3)
    l5 = Conv2D(128, (3, 3), padding='same', activation='relu',
                activity_regularizer=regularizers.l1(10e-10))(l4)
    l6 = MaxPooling2D(padding='same')(l5)
    l7 = Conv2D(256, (3, 3), padding='same', activation='relu',
                activity_regularizer=regularizers.l1(10e-10))(l6)

    l8 = UpSampling2D()(l7)
    l9 = Conv2D(128, (3, 3), padding='same', activation='relu',
                activity_regularizer=regularizers.l1(10e-10))(l8)
    l10 = Conv2D(128, (3, 3), padding='same', activation='relu',
                 activity_regularizer=regularizers.l1(10e-10))(l9)
    l11 = add([l5, l10])
    l12 = UpSampling2D()(l11)
    l13 = Conv2D(64, (3, 3), padding='same', activation='relu',
                 activity_regularizer=regularizers.l1(10e-10))(l12)
    l14 = Conv2D(64, (3, 3), padding='same', activation='relu',
                 activity_regularizer=regularizers.l1(10e-10))(l13)
    l15 = add([l14, l2])
    decoded = Conv2D(3, (3, 3), padding='same', activation='relu',
                     activity_regularizer=regularizers.l1(10e-10))(l15)
    auto_encoder = Model(input_img, decoded)

    print(auto_encoder.summary())

    auto_encoder.compile(optimizer='adadelta', loss='mean_squared_error')

    # auto_encoder_model_name = 'sr.img_net.mse.final_model5.no_patch.weights.best.hdf5'
    auto_encoder_model_name = 'sr.img_net.mse.final_model5.patch.weights.best.hdf5'
    encoder_model_name = 'encoder_weights.hdf5'

    models = os.path.join(APP_ROOT, 'model')

    print("Loading weights")
    auto_encoder.load_weights(os.path.join(models, auto_encoder_model_name))
    encoder.load_weights(os.path.join(models, encoder_model_name))
    print("Loading done...")
    print(" * Model loaded!")

get_model()


def display_prediction(low_res_img_path):
    low_res_img = plt.imread(low_res_img_path)
    print("Shape of image: ",low_res_img.shape)
    image_resized_low_res_img = resize(low_res_img, (256, 256, 3))

    predicted_high_res_img = np.clip(auto_encoder.predict(np.array([image_resized_low_res_img])), 0.0, 1.0)
    encoded_predicted_img = encoder.predict(np.array([image_resized_low_res_img]))

    plot_size = 30

    fig = plt.figure(figsize=(128, 128))

    i = 1
    ax = plt.subplot(plot_size, plot_size, i)
    ax.set_title("Original Low Quality Image")
    plt.imshow(image_resized_low_res_img)

    i += 1
    ax = plt.subplot(plot_size, plot_size, i)
    ax.set_title("Interpolated Image")
    plt.imshow(image_resized_low_res_img, interpolation="bicubic")

    i += 1
    ax = plt.subplot(plot_size, plot_size, i)
    ax.set_title("Encoded Representation")
    plt.imshow(encoded_predicted_img[0].reshape((64 * 64, 256)))

    i += 1
    ax = plt.subplot(plot_size, plot_size, i)
    ax.set_title("Predicted High Quality Image")
    plt.imshow(predicted_high_res_img[0])
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    return fig, extent


@app.route("/")
def index():
    print("starting...")
    # target = os.path.join(APP_ROOT,'images')
    # if os.path.exists(target):
    #     print("Unlike: ",target)
    #     lst = os.listdir(target)
    #     for i in lst:
    #         f = os.path.join(target,i)
    #         os.unlink(f)
    # else:
    #     os.mkdir(target)
    #     print("Dir 'images' created!")
    return render_template("upload.html",img = 'high_res_v_low_res.jpg')


@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images')

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print("file: ", file)
        print("{} is the file name".format(file.filename))
        filename = file.filename

        ext = os.path.splitext(filename)[1].lower()

        if not ext == '.jpg' or ext == '.png':
            return render_template("result.html", image_name="")

        destination = "/".join([target, filename])
        print("Save it to:", destination)
        file.save(destination)

        fig, extent = display_prediction(destination)

        target = os.path.join(APP_ROOT,'prediction')

        if os.path.exists(target):
            lst = os.listdir(target)
            for i in lst:
                f = os.path.join(target,i)
                os.unlink(f)
        else:
            os.mkdir(target)

        print("Dir created! ",target)

        predicted_file1 = 'predicted' + str(time.time()) + ext
        predicted_file2 = 'predicted_full' + str(time.time()) + ext

        output_destination1 = "/".join([target, predicted_file1])
        output_destination2 = "/".join([target, predicted_file2])

        fig.savefig(output_destination1, bbox_inches=extent, dpi=250)
        fig.savefig(output_destination2, bbox_inches='tight', pad_inches=0, dpi=250)

        print("Output saved: ",output_destination1)

    return render_template("result.html", image_names=[predicted_file1,predicted_file2])


#this method works like static folder
@app.route('/prediction/<filename>')
def send_image(filename):
    return send_from_directory("prediction", filename)


@app.route('/progress')
def progress():
    def generate():
        x = 0
        
        while x <= 100:
            yield "data:" + str(x) + "\n\n"
            x = x + 10
            time.sleep(0.6)

    return Response(generate(), mimetype= 'text/event-stream')


if __name__ == "__main__":
    # app.run(port=5001, debug=False)
    app.run(debug=False)
