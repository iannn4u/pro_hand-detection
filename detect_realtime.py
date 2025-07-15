import cv2  # Mengimpor library OpenCV untuk pemrosesan gambar dan video.
import numpy as np  # Mengimpor library NumPy dengan alias 'np' untuk operasi array numerik.
import mediapipe as mp  # Mengimpor library MediaPipe dengan alias 'mp' untuk deteksi tangan.
from tensorflow.keras.models import load_model  # Mengimpor fungsi untuk memuat model Keras yang telah dilatih.
import time  # Mengimpor library 'time' untuk melacak waktu, seperti jeda dan durasi.

model = load_model("models/sign_language_model.h5")  # Memuat model deep learning yang telah dilatih dari file .h5.
labels = {  # Mendefinisikan sebuah dictionary untuk memetakan output numerik model (0-25) ke label huruf (A-Z).
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I",
    9: "J", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R",
    18: "S", 19: "T", 20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z"
}

JEDA_THRESHOLD = 1.5  # Menetapkan waktu jeda (detik) tanpa deteksi tangan yang akan dianggap sebagai spasi.
HOLD_TIME = 1.5  # Menetapkan durasi (detik) sebuah gerakan tangan harus dipertahankan untuk dikonfirmasi sebagai huruf.
MAX_DISPLAY_LENGTH = 10  # Menetapkan jumlah maksimum karakter yang akan ditampilkan di baris teks utama.

current_text = []  # Menginisialisasi list untuk menyimpan kalimat lengkap yang sudah dikonfirmasi.
current_word = []  # Menginisialisasi list untuk menyimpan huruf-huruf dari kata yang sedang diketik.
current_prediction = None  # Variabel untuk menyimpan prediksi huruf saat ini, awalnya kosong.
prediction_start_time = None  # Variabel untuk mencatat waktu kapan sebuah prediksi huruf dimulai.
last_detection_time = time.time()  # Mencatat waktu terakhir kali tangan terdeteksi untuk menghitung jeda.
scroll_offset = 0  # Menginisialisasi posisi awal scroll untuk teks yang panjang.

mp_hands = mp.solutions.hands  # Mengakses modul 'hands' dari dalam library MediaPipe.
hands = mp_hands.Hands(max_num_hands=1)  # Membuat objek detektor tangan yang dikonfigurasi untuk mendeteksi maksimal satu tangan.

cap = cv2.VideoCapture(0)  # Menginisialisasi dan membuka kamera utama (indeks 0).

def get_display_text(full_text, offset, max_length):  # Mendefinisikan fungsi untuk memotong teks agar sesuai dengan layar.
    end_idx = min(offset + max_length, len(full_text))  # Menghitung indeks akhir pemotongan teks agar tidak melebihi panjang teks.
    return full_text[offset:end_idx]  # Mengembalikan potongan list teks sesuai dengan offset dan panjang maksimum.

while cap.isOpened():  # Memulai loop tak terbatas yang akan terus berjalan selama kamera terbuka.
    success, frame = cap.read()  # Membaca satu frame (gambar) dari kamera.
    if not success:  # Memeriksa jika pembacaan frame dari kamera gagal.
        break  # Menghentikan loop jika pembacaan frame gagal.

    frame = cv2.flip(frame, 1)  # Membalik frame secara horizontal (efek cermin).
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Mengonversi format warna frame dari BGR ke RGB untuk MediaPipe.
    results = hands.process(rgb_frame)  # Memproses frame untuk mendeteksi tangan dan mengekstrak landmarks.

    if results.multi_hand_landmarks:  # Memeriksa apakah ada tangan yang terdeteksi di dalam frame.
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark]).flatten()  # Mengubah data landmarks menjadi array NumPy dan meratakannya.
        pred = model.predict(np.expand_dims(landmarks, axis=0), verbose=0)  # Melakukan prediksi huruf menggunakan model pada data landmarks.
        predicted_class = labels[np.argmax(pred)]  # Mengambil label huruf dengan probabilitas tertinggi dari hasil prediksi.

        if current_prediction != predicted_class:  # Memeriksa apakah prediksi saat ini berbeda dari prediksi sebelumnya.
            current_prediction = predicted_class  # Memperbarui prediksi saat ini dengan yang baru.
            prediction_start_time = time.time()  # Mereset timer durasi karena ada prediksi huruf baru.

        if prediction_start_time and (time.time() - prediction_start_time) >= HOLD_TIME:  # Memeriksa apakah huruf telah dipertahankan cukup lama.
            current_word.append(predicted_class)  # Menambahkan huruf yang dikonfirmasi ke kata saat ini.
            prediction_start_time = None  # Mereset timer agar huruf yang sama bisa dideteksi lagi nanti.
            last_detection_time = time.time()  # Memperbarui waktu terakhir kali sebuah huruf dikonfirmasi.

        cv2.putText(frame, f"Prediksi: {predicted_class}", (10, 50),  # Menampilkan teks prediksi huruf saat ini pada frame.
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Mengatur properti teks prediksi.

        hold_progress = min(1.0, (time.time() - prediction_start_time) / HOLD_TIME if prediction_start_time else 0.0)  # Menghitung progres durasi 'hold' (0.0 hingga 1.0).
        cv2.rectangle(frame, (10, 80), (int(10 + 200 * hold_progress), 100), (0, 255, 0), -1)  # Menggambar progress bar yang terisi.
        cv2.rectangle(frame, (10, 80), (210, 100), (255, 255, 255), 1)  # Menggambar bingkai luar dari progress bar.
        cv2.putText(frame, "Hold to confirm", (10, 120),  # Menambahkan teks instruksi di bawah progress bar.
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # Mengatur properti teks instruksi.
    else:  # Blok ini dijalankan jika tidak ada tangan yang terdeteksi.
        current_prediction = None  # Mereset prediksi saat ini menjadi kosong.
        prediction_start_time = None  # Mereset timer durasi.

        if time.time() - last_detection_time > JEDA_THRESHOLD and current_word:  # Memeriksa apakah jeda telah melebihi ambang batas dan ada kata yang sedang diketik.
            current_text.extend(current_word)  # Menambahkan kata yang sudah selesai ke teks utama.
            current_text.append(" ")  # Menambahkan spasi setelah kata tersebut.
            current_word.clear()  # Mengosongkan list kata saat ini untuk kata berikutnya.
            print("Teks saat ini:", "".join(current_text))  # Mencetak teks yang sudah terbentuk ke konsol.

    full_text = current_text + current_word  # Menggabungkan teks yang sudah dikonfirmasi dengan kata yang sedang diketik untuk ditampilkan.
    display_text = get_display_text(full_text, scroll_offset, MAX_DISPLAY_LENGTH)  # Memanggil fungsi untuk memotong teks sesuai tampilan.

    cv2.putText(frame, f"Sedang diketik: {''.join(current_word)}", (10, 150),  # Menampilkan kata yang sedang diketik pada frame.
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)  # Mengatur properti teks kata yang sedang diketik.
    cv2.putText(frame, f"Teks: {''.join(display_text)}", (10, 180),  # Menampilkan teks hasil terjemahan pada frame.
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # Mengatur properti teks hasil terjemahan.

    if len(full_text) > MAX_DISPLAY_LENGTH:  # Memeriksa apakah total teks lebih panjang dari yang bisa ditampilkan.
        cv2.putText(frame, "[...]", (10 + 200, 180),  # Menampilkan indikator '[...]' jika teks bisa di-scroll.
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # Mengatur properti teks indikator.

    key = cv2.waitKey(1) & 0xFF  # Menunggu input keyboard selama 1ms.
    if key == ord('r'):  # Memeriksa jika tombol 'r' ditekan untuk mereset.
        current_text.clear()  # Mengosongkan list teks utama.
        current_word.clear()  # Mengosongkan list kata saat ini.
        current_prediction = None  # Mereset prediksi saat ini.
        prediction_start_time = None  # Mereset timer durasi.
        scroll_offset = 0  # Mereset posisi scroll ke awal.
        last_detection_time = time.time()  # Mereset waktu deteksi terakhir.
        print("Teks di-reset!")  # Mencetak pesan konfirmasi reset ke konsol.
    elif key == ord('q'):  # Memeriksa jika tombol 'q' ditekan untuk keluar.
        break  # Menghentikan loop 'while'.
    elif key == ord('a'):  # Memeriksa jika tombol 'a' ditekan untuk scroll ke kiri.
        scroll_offset = max(0, scroll_offset - 1)  # Mengurangi offset scroll, dengan batas minimal 0.
    elif key == ord('d'):  # Memeriksa jika tombol 'd' ditekan untuk scroll ke kanan.
        scroll_offset = min(len(full_text) - MAX_DISPLAY_LENGTH, scroll_offset + 1)  # Menambah offset scroll, dengan batas maksimal.

    cv2.imshow("Deteksi Bahasa Isyarat", frame)  # Menampilkan frame akhir yang telah dimodifikasi dalam sebuah jendela.

cap.release()  # Melepaskan sumber daya kamera setelah loop berhenti.
cv2.destroyAllWindows()  # Menutup semua jendela yang dibuka oleh OpenCV.