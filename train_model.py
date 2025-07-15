import numpy as np  # Mengimpor library NumPy dengan alias 'np' untuk operasi array numerik.
import tensorflow as tf  # Mengimpor library TensorFlow dengan alias 'tf' untuk machine learning.
from tensorflow.keras.layers import Dense, Dropout  # Mengimpor layer 'Dense' dan 'Dropout' dari Keras untuk membangun model.
from tensorflow.keras.models import Sequential  # Mengimpor kelas 'Sequential' dari Keras untuk membuat model jaringan syaraf tiruan secara berurutan.

data = np.load("models/dataset_landmarks.npz")  # Memuat data latih dan uji dari file .npz yang telah diproses sebelumnya.
X_train, y_train = data["X_train"], data["y_train"]  # Mengekstrak data latih (fitur X dan label y) dari file yang dimuat.
X_test, y_test = data["X_test"], data["y_test"]  # Mengekstrak data uji (fitur X dan label y) dari file yang dimuat.

X_train = X_train.astype(np.float32)  # Mengubah tipe data fitur latih menjadi float32 untuk kompatibilitas dengan TensorFlow.
X_test = X_test.astype(np.float32)  # Mengubah tipe data fitur uji menjadi float32.

model = Sequential([  # Memulai definisi model Sequential, di mana layer ditambahkan secara berurutan.
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  # Menambahkan layer Dense pertama dengan 128 neuron, fungsi aktivasi ReLU, dan bentuk input yang sesuai.
    Dropout(0.5),  # Menambahkan layer Dropout untuk menonaktifkan 50% neuron secara acak guna mencegah overfitting.
    Dense(64, activation='relu'),  # Menambahkan layer Dense kedua dengan 64 neuron dan fungsi aktivasi ReLU.
    Dense(26, activation='softmax')  # Menambahkan layer output dengan 26 neuron (sesuai jumlah kelas huruf) dan aktivasi softmax untuk klasifikasi.
])  # Menutup definisi model Sequential.

model.compile(optimizer='adam',  # Mengonfigurasi proses training model dengan optimizer 'adam'.
              loss='sparse_categorical_crossentropy',  # Menetapkan fungsi loss yang cocok untuk klasifikasi multi-kelas dengan label integer.
              metrics=['accuracy'])  # Menentukan metrik 'accuracy' untuk dipantau selama training dan evaluasi.

history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))  # Memulai proses training model selama 50 epoch menggunakan data latih dan validasi.
model.save("models/sign_language_model.h5")  # Menyimpan model yang telah dilatih ke dalam sebuah file HDF5 (.h5).

loss, accuracy = model.evaluate(X_test, y_test)  # Mengevaluasi performa model pada data uji untuk mendapatkan nilai loss dan akurasi akhir.
print(f"Akurasi Test: {accuracy*100:.2f}%")  # Mencetak akurasi model pada data uji, diformat sebagai persentase dengan dua angka desimal.