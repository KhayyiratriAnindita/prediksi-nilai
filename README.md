# Aplikasi Prediksi Nilai Akhir Semester

Aplikasi web untuk memprediksi nilai akhir semester berdasarkan nilai UTS, UAS, Tugas, dan jam belajar per hari.

## Fitur

- 🔐 Login dan Registrasi
- 📊 Prediksi Nilai Akhir dengan perhitungan otomatis
- 🕐 Riwayat Prediksi
- 👤 Profil Siswa

## Cara Menjalankan

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Jalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

## Cara Menggunakan

1. **Daftar Akun**: Klik tombol "Daftar" dan isi form registrasi
2. **Login**: Masuk dengan email dan password yang telah didaftarkan
3. **Hitung Prediksi**: 
   - Masukkan nilai UTS, UAS, dan Tugas (0-100)
   - Masukkan jam belajar per hari
   - Klik "Hitung Prediksi"
4. **Lihat Riwayat**: Tab "Riwayat" menampilkan semua prediksi yang pernah dihitung
5. **Profil**: Tab "Profil" menampilkan informasi akun Anda

## Rumus Perhitungan

```
Nilai Akhir = (UTS × 30%) + (UAS × 35%) + (Tugas × 25%) + Bonus Jam Belajar

Bonus: Setiap 1 jam belajar = 1 poin (maksimal 10 poin)
```

## Grade

- A: 90-100
- B: 80-89
- C: 70-79
- D: 60-69
- E: 0-59
