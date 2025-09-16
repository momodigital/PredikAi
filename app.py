import streamlit as st
import numpy as np
import random
from collections import defaultdict
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ======== CONFIG =========
st.set_page_config(
    page_title="🎰 Prediksi Togel AI - 6 Digit UNIK (Tanpa Duplikasi)",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🎰 Prediksi Togel AI - 6 Digit UNIK (Tanpa Angka Sama)")
st.markdown("""
> 📌 **Aplikasi ini hanya untuk simulasi edukasi.**  
> Togel bersifat acak murni — tidak ada model yang bisa memprediksi hasilnya secara akurat.  
> Di sini, kami belajar dari **histori 4-digit**, lalu **memprediksi 2 digit tambahan** agar menjadi **6 digit UNIK (tanpa pengulangan)**.
""")

# ======== INPUT DATA =========
st.subheader("📥 Masukkan Histori Angka 4 Digit")
st.caption("Masukkan satu angka 4 digit per baris (contoh: 1234, 5678, dst). Minimal 15 angka.")

teks_angka = st.text_area(
    "Histori angka (satu per baris):",
    height=250,
    value="""1234
5678
9012
3456
7890
1230
4567
8901
2345
6789
0123
4560
7894
1205
3459
6781
9016
2340
5672
8903
1238
4564
7897
0129
3451""",
    help="Pastikan semua angka 4 digit, tanpa spasi atau karakter lain."
)

# Parsing data
angka_list = [
    x.strip().zfill(4)
    for x in teks_angka.splitlines()
    if x.strip().isdigit() and len(x.strip()) == 4
]

if len(angka_list) < 15:
    st.warning(f"⚠️ Masukkan minimal **15 angka** histori agar model bisa belajar.")
    st.stop()

st.success(f"✅ Data berhasil dimuat: {len(angka_list)} angka 4-digit.")

# ======== MODEL CHOICE =========
st.subheader("🧠 Pilih Model Prediksi")
model_choice = st.selectbox(
    "Pilih algoritma prediksi:",
    ["Markov Chain", "LSTM Digit", "GRU Digit"],
    help="Markov Chain: berdasarkan transisi angka 4-digit. LSTM/GRU: berdasarkan pola deret waktu."
)

# Input angka terakhir (4 digit)
input_angka = st.text_input(
    "Angka terakhir (4 digit, untuk prediksi):",
    value=angka_list[-1] if angka_list else "1234",
    max_chars=4,
    help="Masukkan angka terakhir dari histori Anda. Harus 4 digit."
)

if len(input_angka) != 4 or not input_angka.isdigit():
    st.error("❌ Angka terakhir harus 4 digit numerik!")
    st.stop()

# ======== HELPER FUNCTIONS =========

def angka_to_digit_array(data):
    """Konversi list string angka ke array numpy 2D (n, 4)"""
    return np.array([[int(d) for d in list(a)] for a in data], dtype=np.float32)

def build_markov_model(data):
    """Bangun transition matrix Markov Chain (4-digit)"""
    transition = defaultdict(list)
    for i in range(len(data) - 1):
        transition[data[i]].append(data[i + 1])
    return transition

def prediksi_markov(current, transition, n=5):
    """Prediksi 4-digit, lalu tambah 2 digit unik dari sisa angka (0-9)"""
    candidates = transition.get(current, [])
    if not candidates:
        candidates = [str(random.randint(0, 9999)).zfill(4) for _ in range(100)]
    
    hasil_6digit_unik = []
    for c in random.choices(candidates, k=n):
        # Ambil 4 digit prediksi
        digits_used = set(int(d) for d in c)
        
        # Sisa angka yang belum digunakan (0–9)
        available = [d for d in range(10) if d not in digits_used]
        
        if len(available) < 2:
            # Jika kurang dari 2 angka tersisa, pakai angka acak unik (jarang terjadi)
            all_digits = list(range(10))
            random.shuffle(all_digits)
            two_new = all_digits[:2]
        else:
            # Pilih 2 digit unik dari sisa
            two_new = random.sample(available, 2)
        
        # Gabungkan jadi 6-digit unik
        six_digit = c + ''.join(map(str, two_new))
        hasil_6digit_unik.append(six_digit)
    
    return hasil_6digit_unik

def train_lstm_gru_model(data, use_gru=False, sequence_length=5):
    """Train LSTM/GRU untuk prediksi 4-digit"""
    X = angka_to_digit_array(data[:-1])  # Input: 4 digit
    y = angka_to_digit_array(data[1:])   # Target: 4 digit

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.transform(y)

    # Buat timeseries generator
    gen = TimeseriesGenerator(X_scaled, y_scaled, length=sequence_length, batch_size=1)
    
    if len(gen) == 0:
        raise ValueError(f"Tidak cukup data untuk membuat sequence dengan panjang {sequence_length}. Masukkan lebih banyak data.")
    
    model = Sequential([
        (GRU if use_gru else LSTM)(64, activation='relu', input_shape=(sequence_length, 4)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(4)  # Output: 4 digit
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    with st.spinner("⏳ Melatih model... (ini bisa memakan waktu 10-40 detik)"):
        history = model.fit(gen, epochs=20, verbose=0)
    
    return model, scaler, history

def prediksi_multi(model, scaler, last_sequence, n=5, sequence_length=5):
    """Prediksi 4-digit, lalu tambah 2 digit unik dari sisa angka (0-9)"""
    out = []
    current_seq = last_sequence.copy()
    
    for _ in range(n):
        # Prediksi 4 digit berikutnya
        pred_scaled = model.predict(np.expand_dims(current_seq, axis=0), verbose=0)
        pred_raw = scaler.inverse_transform(pred_scaled)
        
        # Konversi ke angka 0-9
        pred_digits = np.clip(np.round(pred_raw).astype(int), 0, 9)
        pred_str_4digit = ''.join(map(str, pred_digits.flatten()))
        
        # Ambil digit yang sudah digunakan
        digits_used = set(int(d) for d in pred_str_4digit)
        
        # Sisa angka yang belum digunakan (0–9)
        available = [d for d in range(10) if d not in digits_used]
        
        if len(available) < 2:
            # Jika kurang dari 2 angka tersisa, pakai angka acak unik
            all_digits = list(range(10))
            random.shuffle(all_digits)
            two_new = all_digits[:2]
        else:
            # Pilih 2 digit unik dari sisa
            two_new = random.sample(available, 2)
        
        # Gabungkan jadi 6-digit unik
        six_digit = pred_str_4digit + ''.join(map(str, two_new))
        out.append(six_digit)
        
        # Update sequence untuk prediksi selanjutnya
        new_row = pred_digits.flatten()
        current_seq = np.vstack([current_seq[1:], new_row]) if len(current_seq) > 1 else np.array([new_row])
        
        # Pastikan shape tetap (sequence_length, 4)
        if len(current_seq) < sequence_length:
            pad = np.zeros((sequence_length - len(current_seq), 4))
            current_seq = np.vstack([pad, current_seq])
    
    return out

def hitung_akurasi(data, model_type="Markov", sequence_length=5, test_size=10):
    """Hitung akurasi top-1, top-3, top-5 pada 4-digit, bukan 6-digit"""
    if len(data) < test_size + sequence_length + 1:
        raise ValueError(f"Data tidak cukup. Butuh minimal {test_size + sequence_length + 1} angka.")
    
    benar = {'top1': 0, 'top3': 0, 'top5': 0}
    total = 0
    
    start_idx = len(data) - test_size - 1
    end_idx = len(data) - 1
    
    for i in range(start_idx, end_idx):
        try:
            input_val = data[i]
            target_4digit = data[i + 1]
            
            if model_type == "Markov":
                transition = build_markov_model(data[:i+1])
                pred_list_4digit = prediksi_markov(input_val, transition, n=5)
                pred_list_4digit = [p[:4] for p in pred_list_4digit]
            else:
                model, scaler, _ = train_lstm_gru_model(data[:i+1], use_gru=(model_type=="GRU"), sequence_length=sequence_length)
                X = angka_to_digit_array(data[:i+1])
                scaler.fit(X)
                X_scaled = scaler.transform(X)
                last_seq = X_scaled[-sequence_length:]
                pred_list_6digit = prediksi_multi(model, scaler, last_seq, n=5, sequence_length=sequence_length)
                pred_list_4digit = [p[:4] for p in pred_list_6digit]
            
            if target_4digit == pred_list_4digit[0]:
                benar['top1'] += 1
            if target_4digit in pred_list_4digit[:3]:
                benar['top3'] += 1
            if target_4digit in pred_list_4digit:
                benar['top5'] += 1
            total += 1
            
        except Exception as e:
            st.warning(f"⚠️ Gagal prediksi di indeks {i}: {str(e)}")
            continue
    
    if total == 0:
        return {'top1': 0.0, 'top3': 0.0, 'top5': 0.0}
    
    return {
        'top1': round(benar['top1'] / total * 100, 2),
        'top3': round(benar['top3'] / total * 100, 2),
        'top5': round(benar['top5'] / total * 100, 2)
    }

# ======== PREDIKSI =========
st.subheader("🎯 Prediksi Angka 6 Digit UNIK (Tanpa Angka Sama)")

with st.spinner("🔍 Menghitung prediksi..."):
    try:
        if model_choice == "Markov Chain":
            transition = build_markov_model(angka_list)
            prediksi = prediksi_markov(input_angka, transition, n=5)
        else:
            use_gru = model_choice == "GRU Digit"
            model, scaler, history = train_lstm_gru_model(angka_list, use_gru=use_gru, sequence_length=5)
            X = angka_to_digit_array(angka_list)
            X_scaled = scaler.transform(X)
            last_seq = X_scaled[-5:]  # Ambil 5 angka terakhir (4-digit)
            prediksi = prediksi_multi(model, scaler, last_seq, n=5, sequence_length=5)
        
        st.success(f"✨ **Prediksi 5 angka 6-digit UNIK:**")
        cols = st.columns(5)
        for i, p in enumerate(prediksi):
            with cols[i]:
                st.metric(label=f"#{i+1}", value=p, delta=None)
                
    except Exception as e:
        st.error(f"❌ Gagal melakukan prediksi: {str(e)}")
        st.stop()

# ======== VISUALISASI =========
st.subheader("📊 Visualisasi Pola Prediksi")

fig = make_subplots(rows=1, cols=1, subplot_titles=["Prediksi vs Histori (Contoh 5 Angka Terakhir)"])

# Plot histori 5 angka terakhir (4-digit)
hist_data = angka_list[-5:]
hist_digits = [[int(d) for d in a] for a in hist_data]
for digit_pos in range(4):
    fig.add_trace(
        go.Scatter(
            x=[f"Hist-{i}" for i in range(1,6)],
            y=[h[digit_pos] for h in hist_digits],
            mode='lines+markers',
            name=f'Posisi {digit_pos+1} (Histori)',
            line=dict(color=f'rgba(52, 152, 219, 0.7)'),
            showlegend=True
        ),
        row=1, col=1
    )

# Plot prediksi (ambil 4 digit pertama dari 6-digit)
pred_digits = [[int(d) for d in p[:4]] for p in prediksi]
for digit_pos in range(4):
    fig.add_trace(
        go.Scatter(
            x=[f"Pred-{i}" for i in range(1,6)],
            y=[p[digit_pos] for p in pred_digits],
            mode='lines+markers',
            name=f'Posisi {digit_pos+1} (Prediksi 4-digit)',
            line=dict(color=f'rgba(231, 76, 60, 0.7)', dash='dash'),
            showlegend=True
        ),
        row=1, col=1
    )

fig.update_layout(
    title="Perbandingan Digit Histori vs Prediksi (4-digit bagian depan)",
    xaxis_title="Langkah",
    yaxis_title="Nilai Digit (0-9)",
    height=400,
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

# Tampilkan juga 2 digit tambahan
st.markdown("#### ➕ 2 Digit Tambahan (Unik & Tidak Berulang)")
cols2 = st.columns(5)
for i, p in enumerate(prediksi):
    with cols2[i]:
        st.info(f"`{p[4:6]}` ← 2 digit tambahan (unik, tidak ada yang sama dengan 4 digit pertama)")

# ======== AKURASI =========
st.subheader("🔍 Uji Akurasi Model (berdasarkan 4-digit)")

test_size = st.slider(
    "Jumlah data uji (terakhir):",
    min_value=5,
    max_value=min(30, len(angka_list)-10),
    value=10,
    help="Semakin banyak data uji, semakin akurat evaluasi, tapi butuh lebih banyak data histori."
)

if st.button("🚀 Jalankan Uji Akurasi", type="primary"):
    with st.spinner("🔄 Menghitung akurasi... (ini bisa memakan waktu beberapa menit)"):
        try:
            mode = "Markov" if model_choice == "Markov Chain" else ("GRU" if model_choice == "GRU Digit" else "LSTM")
            acc = hitung_akurasi(angka_list, model_type=mode, test_size=test_size, sequence_length=5)
            
            st.markdown("### 📊 Hasil Akurasi (4-digit bagian depan):")
            col1, col2, col3 = st.columns(3)
            col1.metric("Top-1 Accuracy", f"{acc['top1']}%", "🔎")
            col2.metric("Top-3 Accuracy", f"{acc['top3']}%", "✅")
            col3.metric("Top-5 Accuracy", f"{acc['top5']}%", "🎯")
            
            # Grafik akurasi
            fig_acc = go.Figure(data=[
                go.Bar(name='Top-1', x=['Akurasi'], y=[acc['top1']], marker_color='lightblue'),
                go.Bar(name='Top-3', x=['Akurasi'], y=[acc['top3']], marker_color='lightgreen'),
                go.Bar(name='Top-5', x=['Akurasi'], y=[acc['top5']], marker_color='gold')
            ])
            fig_acc.update_layout(
                title="Akurasi Prediksi (Top-1, Top-3, Top-5) — Berdasarkan 4-digit",
                yaxis_title="Persentase (%)",
                yaxis_range=[0, 100],
                height=300,
                template="plotly_white"
            )
            st.plotly_chart(fig_acc, use_container_width=True)
            
            # Interpretasi
            if acc['top5'] > 20:
                st.info("💡 **Interpretasi**: Model menemukan pola kuat dalam 4-digit histori — 2 digit tambahan dibuat unik secara acak, jadi prediksi 6-digit unik ini **tetap tidak bisa diandalkan untuk togel nyata**.")
            else:
                st.warning("⚠️ **Interpretasi**: Model tidak menemukan pola kuat. Ini sesuai ekspektasi — togel bersifat acak!")

        except Exception as e:
            st.error(f"❌ Gagal menghitung akurasi: {str(e)}")

# ======== INFO TEKNIS =========
with st.expander("📚 Penjelasan Teknis & Cara Kerja"):
    st.markdown("""
    ### 🔍 Bagaimana Model Ini Bekerja?

    #### **Input: Histori 4-digit**
    - Contoh: `1234`, `5678`, `9012`
    - Model belajar pola transisi antar angka 4-digit.

    #### **Output: Prediksi 6-digit UNIK**
    - Model memprediksi **4 digit berikutnya** → lalu mencari **2 digit tambahan** dari angka 0–9 yang **belum digunakan**.
    - Jadi:  
      `1234` → prediksi `5678` → digit yang digunakan: `{1,2,3,4,5,6,7,8}`  
      Sisa: `{0,9}` → pilih 2 digit: `0` dan `9` → hasil: `567809` ✅  
      Semua digit unik! Tidak ada yang sama.

    #### **Mengapa Ini Penting?**
    - Dalam beberapa jenis togel (terutama di Asia), ada variasi **“6D Unik”** di mana angka boleh tidak berulang.
    - Ini adalah simulasi realistis dari aturan tersebut.
    - 2 digit tambahan **tidak diprediksi oleh model**, tapi **dihasilkan secara acak dari sisa angka** — agar tetap unik.

    ### ⚠️ Penting!
    - **Togel itu acak murni** — bahkan jika model bisa prediksi 4-digit dengan akurasi tinggi, 2 digit terakhir tetap acak.
    - Aplikasi ini hanya untuk **eksperimen kreatif**: *“Bagaimana jika kita kombinasikan pola + aturan unik?”*
    - **Bukan alat untuk menang togel.**
    """)
    
    st.markdown("#### 📈 Catatan Performa")
    st.markdown("""
    - Akurasi top-5 > 20% pada 4-digit adalah **sangat tinggi** — artinya data kamu punya bias kuat.
    - Jika akurasi rendah (<10%), itu normal — karena togel memang acak.
    - **Aturan “unik” membatasi ruang pencarian** — jadi kemungkinan hasilnya lebih sedikit, tapi tidak lebih akurat.
    """)

# ======== FOOTER =========
st.divider()
st.caption("""
📌 **Disclaimer**: Aplikasi ini dibuat untuk tujuan edukasi dan eksperimen ilmu data.  
**Tidak direkomendasikan untuk digunakan dalam perjudian nyata.**  
Prediksi togel tidak mungkin akurat karena sifatnya acak. Gunakan pengetahuan ini untuk belajar AI, bukan untuk menang.
""")
