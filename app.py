from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Muat model ML Anda
model = load_model('model batik.h5')
motives_list = ['Batik Cendrawasih', 'Batik Dayak', 'Batik Ikat Celup', 'Batik Insang', 'Batik Kawung', 'Batik Megamendung', 'Batik Parang', 'Batik Poleng', 'Batik Sekar Jagad', 'Batik Tambal']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Menerima gambar dari aplikasi Flutter
        image = request.files['image']

        # Baca data dari objek FileStorage
        image_data = image.read()

        # Buat objek BytesIO dari data gambar
        img_bytesio = BytesIO(image_data)

        # Buka gambar dan konversi ke mode warna RGB
        img = Image.open(img_bytesio).convert("RGB")

        # Resize gambar sesuai dengan kebutuhan
        img = img.resize((224, 224))

        # Preproses gambar seperti yang Anda lakukan sebelumnya
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Lakukan prediksi dengan model
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        class_name = motives_list[class_index]  # Sesuaikan dengan struktur kelas Anda

        result = {
            "class_name": class_name,
            "confidence": str(prediction[0][class_index])
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route('/predict', methods=['GET']) 
def welcome():
    return "API berhasil diakses."

if __name__ == "__main__":
    app.run(debug=True)
