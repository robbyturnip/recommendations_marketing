# Pada bagian ini, kita mengimpor beberapa library yang akan digunakan dalam kode ini, seperti pickle, numpy, pandas, dan Flask. Library-library ini digunakan untuk membaca model yang telah disimpan, melakukan prediksi, dan membuat aplikasi web.
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Pada bagian ini, kita membaca model yang telah disimpan dalam file 'model.pkl' menggunakan library pickle. Model tersebut kemudian disimpan dalam variabel 'model' dan kelas-kelas yang ada dalam model disimpan dalam variabel 'classes'.
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
    classes = model.classes_

# Fungsi ini digunakan untuk mengambil rekomendasi berdasarkan ID pelanggan yang diberikan. Fungsi ini menggunakan model yang telah dibaca sebelumnya untuk melakukan prediksi probabilitas dan menghasilkan daftar rekomendasi. Daftar rekomendasi kemudian diurutkan berdasarkan probabilitasnya.
def fetch_recommendations(customer_id):
    pred_proba = model.predict_proba(np.array([customer_id]).reshape(-1, 1))
    recomendations = []

    for index, val in enumerate(classes):
        dict_recomendation = {}
        dict_recomendation.setdefault('PRODUCT', val)
        dict_recomendation.setdefault('PROBABILITY', pred_proba[0][index])
        recomendations.append(dict_recomendation)

    recomendations = sorted(recomendations, key=lambda d: d['PROBABILITY'], reverse=True) 

    return recomendations

# Pada bagian ini, kita membuat route untuk halaman utama aplikasi web. Jika metode yang digunakan adalah POST, maka kita mengambil ID pelanggan yang dikirim melalui form dan menggunakan fungsi 'fetch_recommendations' untuk mendapatkan rekomendasi. Hasil rekomendasi kemudian ditampilkan pada halaman utama menggunakan template 'index.html'.
@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        customer_id = request.form.get('customerID')
        recommendations = fetch_recommendations(customer_id)

    return render_template('index.html', recommendations=recommendations)

# Pada bagian ini, kita menjalankan aplikasi web menggunakan Flask dengan mode debug aktif.
if __name__ == '__main__':
    app.run(debug=True)