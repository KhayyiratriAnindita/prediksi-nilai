from datetime import datetime
import time
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import streamlit as st
import psycopg2
import psycopg2.extras

# ===============================
# Konfigurasi halaman
# ===============================
st.set_page_config(
    page_title="Prediksi Nilai Akhir Semester",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===============================
# Cache MODEL (AMAN)
# ===============================
@st.cache_resource
def load_model():
    base_dir = Path(__file__).resolve().parent
    model = joblib.load(base_dir / "model_final.pkl")
    poly = joblib.load(base_dir / "poly_transformer.pkl")
    return model, poly

model, poly = load_model()

# Custom CSS
st.markdown("""
<style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;500;600;700&display=swap');
    
    /* Global styling */
    .stApp {
        background-color: #e0e7ff;
    }
    
    /* Container styling */
    .main-container {
        background-color: white;
        padding: 40px;
        border-radius: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 20px auto;
    }
    
    /* Title styling */
    .main-title {
        font-size: 28px;
        font-weight: bold;
        color: #1e293b;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .subtitle {
        font-size: 14px;
        color: #64748b;
        text-align: center;
        margin-bottom: 30px;
    }
    
    .section-title {
        font-size: 24px;
        font-weight: bold;
        color: #1e293b;
        margin-bottom: 10px;
    }
    
    /* Icon styling */
    .icon-large {
        font-size: 64px;
        text-align: center;
        margin: 20px 0;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #0f172a;
        color: white;
        border-radius: 8px;
        padding: 14px 24px;
        font-size: 15px;
        font-weight: 600;
        border: none;
        width: 100%;
    }
    
    .stButton>button:hover {
        background-color: #1e293b;
    }
    
    /* Input styling */
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #d1d5db;
        background-color: #f9fafb;
        padding: 12px 16px;
    }
    
    /* Info box */
    .info-box {
        background-color: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 20px;
        margin: 20px 0;
    }
    
    .info-title {
        font-size: 14px;
        font-weight: 600;
        color: #1e40af;
        margin-bottom: 10px;
    }
    
    .info-item {
        font-size: 13px;
        color: #1e40af;
        line-height: 1.6;
    }
    
    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 16px;
        text-align: center;
        margin: 20px 0;
    }
    
    .result-score {
        font-size: 48px;
        font-weight: bold;
        margin: 20px 0;
    }
    
    .result-grade {
        font-size: 32px;
        font-weight: bold;
        margin: 10px 0;
    }
    
    /* Header */
    .header {
        background-color: white;
        padding: 20px 30px;
        border-bottom: 1px solid #e5e7eb;
        margin: -70px -80px 30px -80px;
        border-radius: 0;
    }
    
    /* Profile card */
    .profile-card {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .profile-label {
        font-size: 13px;
        color: #64748b;
        font-weight: 500;
        margin-bottom: 5px;
    }
    
    .profile-value {
        font-size: 16px;
        color: #1e293b;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# KONEKSI DATABASE (TIDAK DICACHE)
# ===============================
def get_db_connection():
    return psycopg2.connect(
        host=st.secrets["db"]["host"],
        user=st.secrets["db"]["user"],
        password=st.secrets["db"]["password"],
        dbname=st.secrets["db"]["database"],
        sslmode="require",
        connect_timeout=5
    )

# ===============================
# WAKE UP NEON
# ===============================
def wake_up_db(retries=5):
    for _ in range(retries):
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
            conn.close()
            return True
        except psycopg2.OperationalError:
            time.sleep(2)
    return False

# SIMPAN KE DATABASE
def simpan_ke_db(user_id, presensi, uts, uas, tugas, jam, hasil, grade):
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                """INSERT INTO data_input
                (user_id, presensi, nilai_uts, nilai_uas, nilai_tugas, jam_belajar)
                VALUES (%s,%s,%s,%s,%s,%s)
                RETURNING id_input""",
                (user_id, presensi, uts, uas, tugas, jam)
            )
            id_input = cursor.fetchone()[0]

            cursor.execute(
                """INSERT INTO hasil_prediksi
                (user_id, id_input, nilai_prediksi, grade)
                VALUES (%s,%s,%s,%s)""",
                (user_id, id_input, hasil, grade)
            )
            conn.commit()
    finally:
        if conn:
            conn.close()

# SESSION STATE
if 'page' not in st.session_state:
    st.session_state.page = 'login'
if 'user' not in st.session_state:
    st.session_state.user = None
if 'history' not in st.session_state: 
    st.session_state.history = []
if 'users_db' not in st.session_state: 
    st.session_state.users_db = {}

# NAVIGASI
def go_to_page(page):
    st.session_state.page = page
    st.rerun()

# LOGIN (FIXED)
def login(email, password):
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(
                "SELECT * FROM login_users WHERE email=%s AND password=%s",
                (email, password)
            )
            user = cursor.fetchone()

        if user:
            st.session_state.user = user
            go_to_page('prediction')
            return True
        return False

    except psycopg2.InterfaceError:
        st.error("Koneksi database terputus. Silakan coba lagi.")
        return False

    finally:
        if conn:
            conn.close()

# REGISTER
def register(nama, nis, kelas, email, password):
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                """INSERT INTO login_users
                (nama_lengkap, nis, kelas, email, password)
                VALUES (%s,%s,%s,%s,%s)
                RETURNING id_user""",
                (nama, nis, kelas, email, password)
            )
            user_id = cursor.fetchone()[0]
            conn.commit()

        st.session_state.user = {
            'id_user': user_id,
            'nama_lengkap': nama,
            'nis': nis,
            'kelas': kelas,
            'email': email
        }
        go_to_page('prediction')
        return True

    except Exception as e:
        st.error(f"Error Database: {e}")
        return False

    finally:
        if conn:
            conn.close()

# LOGOUT
def logout():
    st.session_state.user = None
    st.session_state.page = 'login'
    st.rerun()

# LOGIN PAGE
def login_page():
    wake_up_db()

    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="icon-large">🎓</div>', unsafe_allow_html=True)
        st.markdown('<h1 class="main-title">Prediksi Nilai Akhir</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Masuk ke akun Anda untuk memulai</p>', unsafe_allow_html=True)
        
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="siswa@sekolah.com")
            password = st.text_input("Password", type="password", placeholder="Masukkan password")
            
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                submit = st.form_submit_button("Masuk", use_container_width=True)
            with col_btn2:
                register_btn = st.form_submit_button("Daftar", use_container_width=True)
            
            if submit:
                if email and password:
                    # ❌ TIDAK DIUBAH: tetap pakai login(email, password)
                    if login(email, password):
                        st.success("Login berhasil!")
                    else:
                        st.error("Email atau password salah!")
                else:
                    st.warning("Mohon isi semua field!")
            
            if register_btn:
                go_to_page('register')

# REGISTER PAGE
def register_page():
    wake_up_db()

    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="icon-large">🎓</div>', unsafe_allow_html=True)
        st.markdown('<h1 class="main-title">Daftar Akun</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Buat akun baru untuk mulai menggunakan aplikasi</p>', unsafe_allow_html=True)
        
        with st.form("register_form"):
            nama = st.text_input("Nama Lengkap", placeholder="Nama Lengkap")
            nis = st.text_input("NIS (Nomor Induk Siswa)", placeholder="Nomor Induk Siswa")
            kelas = st.text_input("Kelas", placeholder="Kelas")
            email = st.text_input("Email", placeholder="siswa@sekolah.com")
            password = st.text_input("Password", type="password", placeholder="Masukkan password")
            
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                submit = st.form_submit_button("Daftar", use_container_width=True)
            with col_btn2:
                login_btn = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                if nama and nis and kelas and email and password:
                    if register(nama, nis, kelas, email, password):
                        st.success("Registrasi berhasil!")
                    else:
                        st.error("Email sudah terdaftar!")
                else:
                    st.warning("Mohon isi semua field!")
            
            if login_btn:
                go_to_page('login')


# HALAMAN PREDIKSI
def prediction_page():
    # Header
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        st.markdown(f"<h2 style='margin:0; color:#1e293b;'>🎓 Prediksi Nilai Akhir Semester</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#64748b; margin:0;'>Selamat datang, {st.session_state.user['nama_lengkap']}</p>", unsafe_allow_html=True)
    with col_header2:
        if st.button("↪ Keluar", use_container_width=True):
            logout()
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊  Prediksi", "🕐  Riwayat", "👤  Profil"])
    
    # Tab Prediksi
    with tab1:
        st.markdown('<h2 class="section-title">Hitung Prediksi Nilai Akhir</h2>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Masukkan nilai-nilai Anda untuk mendapatkan prediksi nilai akhir semester</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📖 Nilai UTS (0-40)")
            uts = st.number_input("UTS", min_value=0.0, max_value=40.0, value=0.0, step=1.0, label_visibility="collapsed")
            st.caption("Bobot: 10%")
            
            st.markdown("### 📝 Nilai Tugas (0-10)")
            tugas = st.number_input("Tugas", min_value=0.0, max_value=10.0, value=0.0, step=1.0, label_visibility="collapsed")
            st.caption("Bobot: 46%")

            st.markdown("### 📅 Presensi (0-100%)")
            presensi = st.slider("Persentase Presensi", 0, 100, 100, label_visibility="collapsed")
        
        with col2:
            st.markdown("### 📈 Nilai UAS (0-40)")
            uas = st.number_input("UAS", min_value=0.0, max_value=40.0, value=0.0, step=1.0, label_visibility="collapsed")
            st.caption("Bobot: 10%")
            
            st.markdown("### 🕐 Jam Belajar per Hari")
            jam = st.number_input("Jam Belajar", min_value=0.0, max_value=24.0, value=0.0, step=0.5, label_visibility="collapsed")
            st.caption("Bobot: 27%")
        
        # Info box
        st.markdown("""
        <div class="info-box">
            <div class="info-title">Informasi Perhitungan:</div>
            <div class="info-item">• Nilai Akhir = (UTS × 10%) + (UAS × 10%) + (Tugas × 46%) + (Jam Belajar × 27%) + Presensi</div>
            <div class="info-item">• Grade A: 90-100 | B: 80-89 | C: 70-79 | D: 60-69 | E: 0-59</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Hitung Prediksi", use_container_width=True):
            if uts > 0 or uas > 0 or tugas > 0:
                # --- LOGIKA MODEL ML ---
                # 1. Definisikan bonus (agar metric di bawah tidak error)
                bonus = jam # atau sesuaikan: min(jam, 10)

                # 2. Susun data input (Urutan: Presensi, UTS, UAS, Jam Belajar, Tugas)
                input_data = np.array([[presensi, uts, uas, jam, tugas]])
                
                # 3. Transformasi ke Polinomial
                input_poly = poly.transform(input_data)
                
                # 4. Prediksi dengan Model
                prediksi = model.predict(input_poly)
                nilai_akhir = float(prediksi[0])
                
                # Batasi nilai 0-100
                nilai_akhir = max(0, min(nilai_akhir, 100))

                # Tentukan grade
                if nilai_akhir >= 90:
                    grade = "A"
                    status = "Luar Biasa!"
                elif nilai_akhir >= 80:
                    grade = "B"
                    status = "Bagus!"
                elif nilai_akhir >= 70:
                    grade = "C"
                    status = "Cukup Baik"
                elif nilai_akhir >= 60:
                    grade = "D"
                    status = "Perlu Peningkatan"
                else:
                    grade = "E"
                    status = "Harus Belajar Lebih Giat"
                
                # Simpan ke riwayat
                st.session_state.history.append({
                    'tanggal': datetime.now().strftime("%d/%m/%Y %H:%M"),
                    'uts': uts,
                    'uas': uas,
                    'tugas': tugas,
                    'jam_belajar': jam,
                    'nilai_akhir': nilai_akhir,
                    'grade': grade
                })

                # TAMBAHKAN INI: Simpan ke MySQL permanen
                try:
                    user_id_sekarang = st.session_state.user.get('id_user', 1) # Kita asumsikan user_id didapat dari database saat login (lihat poin 2)
                    simpan_ke_db(user_id_sekarang, presensi, uts, uas, tugas, jam, nilai_akhir, grade)
                    st.toast("Data berhasil disimpan ke Database!", icon="💾")
                except Exception as e:
                    st.error(f"Gagal simpan ke database: {e}")
                
                # Tampilkan hasil
                st.markdown(f"""
                <div class="result-card">
                    <h3 style="margin:0;">Hasil Prediksi Nilai Akhir</h3>
                    <div class="result-score">{nilai_akhir:.2f}</div>
                    <div class="result-grade">Grade: {grade}</div>
                    <p style="font-size:18px; margin:10px 0 0 0;">{status}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Detail breakdown
                col_detail1, col_detail2, col_detail3, col_detail4 = st.columns(4)
                with col_detail1:
                    st.metric("UTS (10%)", f"{uts * 0.10:.2f}")
                with col_detail2:
                    st.metric("UAS (10%)", f"{uas * 0.10:.2f}")
                with col_detail3:
                    st.metric("Tugas (46%)", f"{tugas * 0.46:.2f}")
                with col_detail4:
                    st.metric("Jam Belajar (27%)", f"{jam * 0.27:.2f}")
            else:
                st.warning("Mohon isi minimal satu nilai!")
    
    # Tab Riwayat
    with tab2:
        st.markdown('<h2 class="section-title">Riwayat Prediksi</h2>', unsafe_allow_html=True)
        conn = get_db_connection()
        query = """
            SELECT h.tanggal_prediksi AS Tanggal, d.nilai_uts AS UTS, d.nilai_uas AS UAS, 
                   d.nilai_tugas AS Tugas, d.jam_belajar AS Jam, 
                   h.nilai_prediksi AS "Nilai Akhir", h.grade AS Grade
            FROM hasil_prediksi h
            JOIN data_input d ON h.id_input = d.id_input
            WHERE h.user_id = %s
            ORDER BY h.tanggal_prediksi DESC
        """
        # Gunakan pandas dengan cara ini agar lebih stabil
        df_riwayat = pd.read_sql(query, conn, params=(st.session_state.user['id_user'],))

        if not df_riwayat.empty:
            df_riwayat["nilai_akhir"] = df_riwayat["nilai_akhir"].astype(float)
            df_riwayat["uts"] = df_riwayat["uts"].astype(float)
            df_riwayat["uas"] = df_riwayat["uas"].astype(float)
            df_riwayat["tugas"] = df_riwayat["tugas"].astype(float)
            df_riwayat["jam"] = df_riwayat["jam"].astype(float)

        if df_riwayat.empty:
            st.info("Belum ada riwayat prediksi untuk akun ini.")
        else:
            st.dataframe(df_riwayat, use_container_width=True, hide_index=True)
    
    # Tab Profil
    with tab3:
        st.markdown('<h2 class="section-title">Profil Siswa</h2>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Informasi akun Anda</p>', unsafe_allow_html=True)
        
        col_profile1, col_profile2 = st.columns(2)
        
        with col_profile1:
            st.markdown(f"""
            <div class="profile-card">
                <div class="profile-label">Nama Lengkap</div>
                <div class="profile-value">{st.session_state.user['nama_lengkap']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="profile-card">
                <div class="profile-label">Kelas</div>
                <div class="profile-value">{st.session_state.user['kelas']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_profile2:
            st.markdown(f"""
            <div class="profile-card">
                <div class="profile-label">NIS</div>
                <div class="profile-value">{st.session_state.user['nis']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="profile-card">
                <div class="profile-label">Email</div>
                <div class="profile-value">{st.session_state.user['email']}</div>
            </div>
            """, unsafe_allow_html=True)

# ROUTER
if st.session_state.page == 'login':
    login_page()
elif st.session_state.page == 'register':
    register_page()
elif st.session_state.page == 'prediction':
    if st.session_state.user:
        prediction_page()
    else:
        go_to_page('login')
