from numpy.core.fromnumeric import shape
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess_input
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess_input
from tensorflow.keras.applications.nasnet import preprocess_input as nasnet_preprocess_input
from flask import Flask, request, jsonify
from PIL import Image
import io
import os

MODEL_PATH = os.path.join('static', 'trained model')

app: Flask = Flask(__name__)
app.config['UPLOAD_MODEL'] = MODEL_PATH

threshold = 0.5
model_list = {'effnet': {'filename': 'EffNet.h5', 'image_size': 512},
              'xception': {'filename': 'Xception.h5', 'image_size': 512},
              'nasnetmobile': {'filename': 'NasNetMobile.h5', 'image_size': 224},
              }


def prepeocess(image, size):
    import numpy as np
    img = image.read()
    img = Image.open(io.BytesIO(img))
    img = img.convert("RGB")
    img = img.resize((size, size))
    img = np.array(img)
    return np.expand_dims(img, axis=0)


def let_predict(model_name, image):
    if model_name == 'effnet':
        img = effnet_preprocess_input(image)
        return model_effnet.predict(img)
    elif model_name == 'xception':
        img = xception_preprocess_input(image)
        return model_xception.predict(img)
    elif model_name == 'nasnetmobile':
        img = nasnet_preprocess_input(image)
        return model_nasnetmobile.predict(img)


def check_class(result):
    if result[0][0] < threshold:
        prob = 100-result[0][0]*100
        return 'defect', prob
    else:
        return 'Ok', result[0][0]*100


model_effnet = load_model(os.path.join(
    MODEL_PATH, model_list['effnet']['filename']))
print('model_effnet uploaded.')

model_xception = load_model(os.path.join(
    MODEL_PATH, model_list['xception']['filename']))
print('model_xception uploaded.')

model_nasnetmobile = load_model(os.path.join(
    MODEL_PATH, model_list['nasnetmobile']['filename']))
print('model_nasnetmobile uploaded.')


@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return "This server is healthy"


@app.route('/predict/<string:model_name>', methods=['POST'])
def predict(model_name):
    if model_name.lower() in model_list:
        model_name = model_name.lower()
        predicted = []
        for image in request.files.getlist("image"):
            filename = image.filename
            img = prepeocess(image,
                             model_list[model_name]["image_size"])
            result = let_predict(model_name, img)
            predict, prob = check_class(result)
            predicted.append(
                {filename: {"predict": predict, "probability": prob}})
            print(f"{filename} : prediced as {predict} with probability {prob}%")
        return jsonify(predicted)
    else:
        return "This model is not avaliable."


def main():
    app.run(host="0.0.0.0", port="5000", debug=True)


if __name__ == "__main__":
    main()
