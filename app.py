import streamlit as st
import pandas as pd
from datetime import datetime

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Nilai Akhir Semester",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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

# Inisialisasi session state
if 'page' not in st.session_state:
    st.session_state.page = 'login'
if 'user' not in st.session_state:
    st.session_state.user = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'users_db' not in st.session_state:
    st.session_state.users_db = {}

# Fungsi navigasi
def go_to_page(page_name):
    st.session_state.page = page_name
    st.rerun()

# Fungsi login
def login(email, password):
    if email in st.session_state.users_db:
        if st.session_state.users_db[email]['password'] == password:
            st.session_state.user = st.session_state.users_db[email]
            go_to_page('prediction')
            return True
    return False

# Fungsi register
def register(nama, nis, kelas, email, password):
    if email in st.session_state.users_db:
        return False
    st.session_state.users_db[email] = {
        'nama': nama,
        'nis': nis,
        'kelas': kelas,
        'email': email,
        'password': password
    }
    st.session_state.user = st.session_state.users_db[email]
    go_to_page('prediction')
    return True

# Fungsi logout
def logout():
    st.session_state.user = None
    st.session_state.page = 'login'
    st.rerun()

# Halaman Login
def login_page():
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
                    if login(email, password):
                        st.success("Login berhasil!")
                    else:
                        st.error("Email atau password salah!")
                else:
                    st.warning("Mohon isi semua field!")
            
            if register_btn:
                go_to_page('register')

# Halaman Register
def register_page():
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

# Halaman Prediksi
def prediction_page():
    # Header
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        st.markdown(f"<h2 style='margin:0; color:#1e293b;'>🎓 Prediksi Nilai Akhir Semester</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#64748b; margin:0;'>Selamat datang, {st.session_state.user['nama']}</p>", unsafe_allow_html=True)
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
            st.markdown("### 📖 Nilai UTS (0-100)")
            uts = st.number_input("UTS", min_value=0.0, max_value=100.0, value=0.0, step=1.0, label_visibility="collapsed")
            st.caption("Bobot: 30%")
            
            st.markdown("### 📝 Nilai Tugas (0-100)")
            tugas = st.number_input("Tugas", min_value=0.0, max_value=100.0, value=0.0, step=1.0, label_visibility="collapsed")
            st.caption("Bobot: 25%")
        
        with col2:
            st.markdown("### 📈 Nilai UAS (0-100)")
            uas = st.number_input("UAS", min_value=0.0, max_value=100.0, value=0.0, step=1.0, label_visibility="collapsed")
            st.caption("Bobot: 35%")
            
            st.markdown("### 🕐 Jam Belajar per Hari")
            jam = st.number_input("Jam Belajar", min_value=0.0, max_value=24.0, value=0.0, step=0.5, label_visibility="collapsed")
            st.caption("Bonus maksimal: 10 poin")
        
        # Info box
        st.markdown("""
        <div class="info-box">
            <div class="info-title">Informasi Perhitungan:</div>
            <div class="info-item">• Nilai Akhir = (UTS × 30%) + (UAS × 35%) + (Tugas × 25%) + Bonus Jam Belajar</div>
            <div class="info-item">• Setiap 1 jam belajar per hari = 1 poin bonus (maksimal 10 poin)</div>
            <div class="info-item">• Grade A: 90-100 | B: 80-89 | C: 70-79 | D: 60-69 | E: 0-59</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Hitung Prediksi", use_container_width=True):
            if uts > 0 or uas > 0 or tugas > 0:
                # Hitung nilai akhir
                bonus = min(jam, 10)
                nilai_akhir = (uts * 0.30) + (uas * 0.35) + (tugas * 0.25) + bonus
                nilai_akhir = min(nilai_akhir, 100)  # Maksimal 100
                
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
                    st.metric("UTS (30%)", f"{uts * 0.30:.2f}")
                with col_detail2:
                    st.metric("UAS (35%)", f"{uas * 0.35:.2f}")
                with col_detail3:
                    st.metric("Tugas (25%)", f"{tugas * 0.25:.2f}")
                with col_detail4:
                    st.metric("Bonus", f"{bonus:.2f}")
            else:
                st.warning("Mohon isi minimal satu nilai!")
    
    # Tab Riwayat
    with tab2:
        st.markdown('<h2 class="section-title">Riwayat Prediksi</h2>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Daftar prediksi nilai yang pernah Anda hitung</p>', unsafe_allow_html=True)
        
        if len(st.session_state.history) == 0:
            st.markdown('<div class="icon-large">📋</div>', unsafe_allow_html=True)
            st.markdown('<h3 style="text-align:center; color:#64748b;">Belum ada riwayat prediksi</h3>', unsafe_allow_html=True)
            st.markdown('<p style="text-align:center; color:#94a3b8;">Mulai hitung prediksi nilai Anda di tab Prediksi</p>', unsafe_allow_html=True)
        else:
            # Tampilkan sebagai tabel
            df = pd.DataFrame(st.session_state.history)
            df = df[['tanggal', 'uts', 'uas', 'tugas', 'jam_belajar', 'nilai_akhir', 'grade']]
            df.columns = ['Tanggal', 'UTS', 'UAS', 'Tugas', 'Jam Belajar', 'Nilai Akhir', 'Grade']
            
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Tab Profil
    with tab3:
        st.markdown('<h2 class="section-title">Profil Siswa</h2>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Informasi akun Anda</p>', unsafe_allow_html=True)
        
        col_profile1, col_profile2 = st.columns(2)
        
        with col_profile1:
            st.markdown(f"""
            <div class="profile-card">
                <div class="profile-label">Nama Lengkap</div>
                <div class="profile-value">{st.session_state.user['nama']}</div>
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

# Router
if st.session_state.page == 'login':
    login_page()
elif st.session_state.page == 'register':
    register_page()
elif st.session_state.page == 'prediction':
    if st.session_state.user:
        prediction_page()
    else:
        go_to_page('login')
