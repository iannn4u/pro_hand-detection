import os  # Mengimpor library 'os' untuk berinteraksi dengan sistem operasi, seperti manajemen file dan direktori.
import cv2  # Mengimpor library OpenCV ('cv2') untuk fungsi pemrosesan gambar, seperti membaca file gambar.
import numpy as np  # Mengimpor library NumPy dengan alias 'np' untuk operasi numerik, terutama array.
import mediapipe as mp  # Mengimpor library MediaPipe dengan alias 'mp' untuk deteksi tangan dan landmarks.
from sklearn.model_selection import train_test_split  # Mengimpor fungsi 'train_test_split' dari Scikit-learn untuk membagi dataset.

mp_hands = mp.solutions.hands  # Mengakses modul 'hands' dari dalam library MediaPipe.
hands = mp_hands.Hands(  # Membuat objek detektor tangan dari MediaPipe.
    static_image_mode=True,  # Mengatur mode deteksi untuk gambar diam, bukan video.
    max_num_hands=1,  # Mengatur agar detektor hanya mencari dan memproses maksimal satu tangan per gambar.
    min_detection_confidence=0.7  # Menetapkan ambang batas kepercayaan deteksi minimal sebesar 70%.
)  # Menutup pembuatan objek detektor tangan.

def extract_hand_landmarks(image_path):  # Mendefinisikan fungsi untuk mengekstrak landmarks dari satu gambar.
    image = cv2.imread(image_path)  # Membaca file gambar dari path yang diberikan menggunakan OpenCV.
    if image is None:  # Memeriksa apakah gambar gagal dibaca (misalnya file rusak atau tidak ada).
        print(f"⚠️ Gambar rusak/tidak terbaca: {image_path}")  # Mencetak pesan peringatan jika gambar rusak.
        os.remove(image_path)  # Menghapus file gambar yang rusak dari sistem.
        return None  # Mengembalikan nilai 'None' untuk menandakan kegagalan.

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Mengonversi format warna gambar dari BGR (OpenCV) ke RGB (MediaPipe).
    results = hands.process(image)  # Memproses gambar dengan detektor MediaPipe untuk mencari tangan.

    if not results.multi_hand_landmarks:  # Memeriksa apakah tidak ada tangan yang terdeteksi di dalam gambar.
        print(f"❌ Tangan tidak terdeteksi di: {image_path}")  # Mencetak pesan error jika tidak ada tangan yang ditemukan.
        os.remove(image_path)  # Menghapus file gambar yang tidak mengandung deteksi tangan.
        return None  # Mengembalikan nilai 'None' untuk menandakan kegagalan.

    landmarks = results.multi_hand_landmarks[0]  # Mengambil data landmarks dari tangan pertama yang terdeteksi.
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()  # Mengubah data landmarks (koordinat x, y, z) menjadi array NumPy dan meratakannya (flatten).

def process_dataset(dataset_dir):  # Mendefinisikan fungsi untuk memproses seluruh dataset gambar.
    X, y = [], []  # Menginisialisasi dua list kosong, X untuk data fitur (landmarks) dan y untuk label.
    deleted_files = 0  # Menginisialisasi penghitung untuk jumlah file yang dihapus.

    for label, class_name in enumerate(sorted(os.listdir(dataset_dir))):  # Memulai loop untuk setiap folder kelas (misal: 'A', 'B') dalam direktori dataset.
        class_dir = os.path.join(dataset_dir, class_name)  # Membuat path lengkap menuju direktori kelas saat ini.
        if not os.path.isdir(class_dir):  # Memeriksa apakah item tersebut adalah direktori (bukan file).
            continue  # Melanjutkan ke iterasi berikutnya jika bukan direktori.

        print(f"\n🔍 Memproses kelas: {class_name} (Label: {label})")  # Mencetak status kelas mana yang sedang diproses.
        class_images = os.listdir(class_dir)  # Mendapatkan daftar semua nama file gambar di dalam direktori kelas.
        total_in_class = len(class_images)  # Menghitung jumlah total gambar dalam kelas tersebut.
        processed = 0  # Menginisialisasi penghitung untuk gambar yang berhasil diproses di kelas ini.

        for image_name in class_images:  # Memulai loop untuk setiap gambar di dalam direktori kelas.
            image_path = os.path.join(class_dir, image_name)  # Membuat path lengkap menuju file gambar saat ini.
            landmarks = extract_hand_landmarks(image_path)  # Memanggil fungsi untuk mengekstrak landmarks dari gambar.

            if landmarks is not None:  # Memeriksa apakah ekstraksi landmarks berhasil (tidak mengembalikan None).
                X.append(landmarks)  # Menambahkan data landmarks yang berhasil diekstrak ke list X.
                y.append(label)  # Menambahkan label kelas yang sesuai ke list y.
                processed += 1  # Menambah hitungan gambar yang berhasil diproses.
            else:  # Blok ini dijalankan jika ekstraksi landmarks gagal.
                deleted_files += 1  # Menambah hitungan file yang dihapus.

        print(f"✅ Berhasil: {processed}/{total_in_class} gambar")  # Mencetak ringkasan jumlah gambar yang berhasil diproses untuk kelas tersebut.

    return X, y, deleted_files  # Mengembalikan data X, label y, dan jumlah file yang dihapus setelah semua kelas diproses.

if __name__ == "__main__":  # Blok kondisional yang memastikan kode di dalamnya hanya berjalan jika skrip dieksekusi langsung.
    dataset_dir = "dataset"  # Mendefinisikan nama direktori dataset utama.

    print("🔄 Memulai ekstraksi landmark...")  # Mencetak pesan bahwa proses dimulai.
    X, y, deleted_files = process_dataset(dataset_dir)  # Memanggil fungsi utama untuk memproses dataset dan menyimpan hasilnya.

    if len(X) < 2:  # Memeriksa apakah jumlah sampel yang valid kurang dari 2.
        raise ValueError("\n❌ Dataset terlalu kecil setelah pembersihan. Periksa kualitas gambar!")  # Memberikan error jika dataset terlalu kecil untuk diproses lebih lanjut.

    print(f"\n📊 Hasil akhir:")  # Mencetak header untuk ringkasan hasil akhir.
    print(f"- Total sampel valid: {len(X)}")  # Mencetak jumlah total sampel data yang valid.
    print(f"- Total gambar dihapus: {deleted_files}")  # Mencetak jumlah total gambar yang dihapus selama proses.

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Membagi data menjadi set pelatihan (80%) dan pengujian (20%).
    np.savez("models/dataset_landmarks.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)  # Menyimpan data yang sudah dibagi ke dalam satu file .npz.
    print("\n💾 Data berhasil disimpan di models/dataset_landmarks.npz")  # Mencetak pesan konfirmasi bahwa data telah disimpan.