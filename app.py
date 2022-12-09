# Import library 
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, render_template, request

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing import image

# Flask object
app = Flask(__name__)

# Load the Machine Learning Model
model = tf.keras.models.load_model('model/model_batikhug.h5', custom_objects={'KerasLayer':hub.KerasLayer})

# Path to save upload images
target_img = os.path.join(os.getcwd(), 'static/image_upload')

# Function to allow files format
ALLOW_EXTENSION = {'jpg, jpeg, png'}
def allowed_file(filename):
  return '.' in filename and filename.rsplit('.', 1)[1]

# Function to read the images
def read_image(filename):
  img = load_img(filename, target_size=(150, 150))
  images = image.img_to_array(img)
  images /= 255
  images = np.expand_dims(images, axis=0)
  return images

# Server test function
@app.route('/')
def index():
  return render_template('index.html')

@app.route('/yogyakarta')
def yogyakarta():
  return render_template('yogyakarta.html')

@app.route('/pekalongan')
def pekalongan():
  return render_template('pekalongan.html')

@app.route('/solo')
def solo():
  return render_template('solo.html')

@app.route('/cirebon')
def cirebon():
  return render_template('cirebon.html')

@app.route('/surabaya')
def surabaya():
  return render_template('surabaya.html')

# Route predict images
@app.route('/predict', methods=['GET', 'POST'])
def predict():
  if request.method == 'POST':
    file = request.files['file']
    if file and allowed_file(file.filename): 
      filename = file.filename
      file_path = os.path.join('static/image_upload', filename)
      file.save(file_path)
      img = read_image(file_path) 
      class_prediction = model.predict(img) 
      classes_x = np.argmax(class_prediction,axis=1)
      max_value = np.max(class_prediction)

      if classes_x == 0:
        name = "Batik Betawi" + " ({:.0%})".format(max_value)
        origin = "Derived from the Jakarta"
        filosifi = "Betawi Batik is a traditional craft of the people of Jakarta. Its manufacture began in the 19th century. The initial motif followed the pattern of batik in the north coast of Java Island, namely the theme of the coast. The pattern of Betawi batik was influenced by Chinese culture. Betawi batik motifs use Middle Eastern calligraphy. In addition, Betawi batik uses motifs developed from triangular shapes."
      elif classes_x == 1:
        name = "Batik Cendrawasih" + " ({:.0%})".format(max_value)
        origin = "Derived from the Papua"
        filosifi = "Papuan batik is a typical clothing from the Papua region. This is also developing apart from the Papua region itself. At first, Papuan batik was heavily influenced by batik styles from Pekalongan because business calculations were more profitable. Batik motifs from Papua were produced in Pekalongan, then sent to Papua and traded. as Papuan batik. Papuan batik began to develop around 1985, the motifs that developed were a blend of the two cultures between Papua and Pekalongan. Pekalongan is a Javanese ethnic group that produces batik, combined with Papuan ethnicity, which is rich in ornaments developed as batik motifs. Papuan batik, the result of a combination of these two cultures, is also known by another nickname, namely: Port Numbay Batik. Papuan batik has its own uniqueness from the aspect of its motifs, because it was developed from the cultural richness and exotic natural uniqueness of Papua."
      elif classes_x == 2:
        name = "Batik Kawung" + " ({:.0%})".format(max_value)
        origin = "Derived from the Yogyakarta"
        filosifi = "Kawung batik is a batik motif whose shape is in the form of spheres resembling kawung fruit (a type of coconut or sometimes also considered as aren or fro) arranged neatly geometrically. Sometimes, this motif is also interpreted as a picture of a lotus (lotus) flower with four broken petals. Lotus is a flower that symbolizes longevity and purity."
      elif classes_x == 3:
        name = "Batik Megamendung" + " ({:.0%})".format(max_value)
        origin = "Derived from the Cirebon"
        filosifi = "History Motif Batik Megamendung: \n The most well-known and iconic Cirebon batik motif is the Megamendung motif. This motif symbolizes the rain-carrying cloud as a symbol of fertility and life-giving. The history of this motif is related to the history of the arrival of the Chinese in Cirebon, namely Sunan Gunung Jati who married a Chinese woman named Ong Tie. This motif has very good color gradations with the coloring process being carried out more than three times."
      elif classes_x == 4:
        name = "Batik Parang" + " ({:.0%})".format(max_value)
        origin = "Derived from the Solo"
        filosifi = "Parang batik has a high meaning and has great value in its philosophy. This batik motif from Java is the oldest basic batik motif. This parang batik has the meaning of admonition to never give up, like the ocean waves that never stop moving. Parang batik also depicts a bond that is never broken, both in terms of efforts to improve oneself, efforts to fight for welfare, and forms of family ties."
        if max_value < 0.5 :
          name = "The system cannot recognize the image"
          origin = ""
          filosifi = ""
      return render_template('predict.html', name = name, origin = origin, filosifi = filosifi, user_image = file_path)
    else:
      return "Unable to read the file. Please check file extension"

if __name__ == '__main__':
  app.run(port=12000, debug=True)