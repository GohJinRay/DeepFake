from flask import *  # render_template, request
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
import librosa 
# Create Flask
app = Flask(__name__)

# Set max file size for the images as 10mb
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# Only allow png, jpg, and jpeg files (image)
ALLOWED_EXTENSIONS_IMG = ['png', 'jpg', 'jpeg']


# Only allow .mp3, .wav
ALLOWED_EXTENSIONS_VOICE = ['mp3', 'wav']

def allowed_file_types_img(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_IMG

def allowed_file_types_voice(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_VOICE


# Set global default
def init():
    pass  # No need for graph with eager execution in TensorFlow 2.x

# Loading and processing the image input
def read_image(filename):
    # Load image
    img = load_img(filename, target_size=(128, 128))  # Adjust target size as per your model

    # Convert to array
    img = img_to_array(img)

    # Reshape for the model
    img = img.reshape(1, 128, 128, 3)

    # Rescale the image
    img = img.astype('float32')
    img = img / 255.0  # Normalize
    return img

# Audio feature extraction using librosa
def extract_audio_features(filename, max_length = 1000):
    # Load audio file using librosa
    features = []
    audio, _ = librosa.load(filename, sr = 16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=160000, n_mfcc=100)
    if mfccs.shape[1] < max_length:
        mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :max_length]
    features.append(mfccs)
    return np.array(features)


# Setting the homepage
@app.route('/', methods=['GET', 'POST'])
def home():
    try:
        return render_template('home.html')
    except Exception as e:
        return f"An error occurred while rendering the template: {str(e)}"

# Setting the image upload page
@app.route('/img_upload', methods=['GET', 'POST'])
def img_upload():
    try:
        return render_template('img_upload.html')
    except Exception as e:
        return f"An error occurred while rendering the template: {str(e)}"

# Setting the image upload page
@app.route('/vid_upload', methods=['GET', 'POST'])
def vid_upload():
    try:
        return render_template('vid_upload.html')
    except Exception as e:
        return f"An error occurred while rendering the template: {str(e)}"

# Setting the voice upload page
@app.route('/voice_upload', methods=['GET', 'POST'])
def voice_upload():
    try:
        return render_template('voice_upload.html')
    except Exception as e:
        return f"An error occurred while rendering the template: {str(e)}"

# Predict the class of the image
@app.route('/predict_img', methods=['GET', 'POST'])
def predict_img():
    if request.method == 'POST':
        files = request.files.getlist('file')  # Get the list of files
        predictions = []
        try:
            for file in files:
                if file and allowed_file_types_img(file.filename):
                    filename = file.filename
                    file_path = os.path.join('static/images', filename)
                    file.save(file_path)

                    # Load and preprocess image
                    img = read_image(file_path)

                    # Predict the class of an image
                    model_path = os.path.join(os.getcwd(), 'models', 'deepfake_imagedetector_shorterdataset.keras')
                    model = keras.models.load_model(model_path)
                    # Perform the prediction
                    class_prediction = model.predict(img)
                    print(f"Raw prediction: {class_prediction[0]}") 
                    # Handle the prediction result
                    if class_prediction[0] > 0.5:  # Assuming the output is a probability (e.g., sigmoid output)
                        product = "Real Image"
                    else:
                        product = "Deepfake Image"
                    
                    predictions.append({
                        'product': product,
                        'user_image': file_path,
                        'raw_prediction': class_prediction[0],
                        'filename': filename
                    })

            return render_template('predict_img.html', predictions=predictions)

        except Exception as e:
            return f"An error occurred while processing the files: {str(e)}"
    
    return render_template('predict_img.html')

# Predict the class of the image
@app.route('/predict_vid', methods=['GET', 'POST'])
def predict_vid():
    if request.method == 'POST':
        files = request.files.getlist('file')  # Get the list of files
        predictions = []
        try:
            for file in files:
                if file and allowed_file_types_img(file.filename):
                    filename = file.filename
                    file_path = os.path.join('static/videos', filename)
                    file.save(file_path)

                    # Load and preprocess video
                    vid = read_image(file_path)

                    # Predict the class of a video
                    model_path = os.path.join(os.getcwd(), 'models', 'xxxx.keras')
                    model = keras.models.load_model(model_path)
                    # Perform the prediction
                    class_prediction = model.predict(vid)
                    print(f"Raw prediction: {class_prediction[0]}") 
                    # Handle the prediction result
                    if class_prediction[0] > 0.5:  # sigmoid output
                        product = "Real Video"
                    else:
                        product = "Deepfake Video"
                    
                    predictions.append({
                        'product': product,
                        'user_image': file_path,
                        'raw_prediction': class_prediction[0],
                        'filename': filename
                    })

            return render_template('predict_vid.html', predictions=predictions)

        except Exception as e:
            return f"An error occurred while processing the files: {str(e)}"
    
    return render_template('predict_vid.html')

# Predict the class of the voice
@app.route('/predict_voice', methods=['GET', 'POST'])
def predict_voice():
    app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024 # 1Gb size
    if request.method == 'POST':
        files = request.files.getlist('file')  # Get the list of files
        predictions = []
        try:
            for file in files:
                if file and allowed_file_types_voice(file.filename):
                    filename = file.filename
                    file_path = os.path.join('static/voices', filename)
                    file.save(file_path)

                    # Load and preprocess voice
                    voice_features = extract_audio_features(file_path)

                    # Predict the class of an voice
                    model_path = os.path.join(os.getcwd(), 'models', 'audio_epoch500.keras')
                    model = keras.models.load_model(model_path)
                    # Perform the prediction
                    class_prediction = model.predict(voice_features)
                    print(f"Raw prediction: {class_prediction}") 
                    # Handle the prediction result
                    if class_prediction[0] > 0.5:  # sigmoid output
                        product = "Real Voice"
                    else:
                        product = "Deepfake Voice"
                    
                    predictions.append({
                        'product': product,
                        'user_audio': file_path,
                        'filename': filename,
                        'raw_prediction': class_prediction[0]
                    })

            return render_template('predict_voice.html', predictions=predictions)

        except Exception as e:
            return f"An error occurred while processing the files: {str(e)}"
    
    return render_template('predict_voice.html')

if __name__ == '__main__':
    init()
    app.run(host="0.0.0.0", port=3000, debug=True)
