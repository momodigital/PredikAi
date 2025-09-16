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
    page_title="🎰 Prediksi Togel AI - 6 Digit (Markov/LSTM/GRU)",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🎰 Prediksi Togel AI - Angka 6 Digit")
st.markdown("""
> 📌 **Aplikasi ini hanya untuk simulasi edukasi.**  
> Togel 6 digit (SGP/HK) bersifat acak murni — tidak ada model yang bisa memprediksi hasilnya secara akurat.  
> Gunakan aplikasi ini untuk belajar tentang *Markov Chain*, *LSTM*, dan *GRU* dalam konteks deret waktu.
""")

# ======== INPUT DATA =========
st.subheader("📥 Masukkan Histori Angka 6 Digit")
st.caption("Masukkan satu angka 6 digit per baris (contoh: 123456, 789012, dst). Minimal 15 angka.")

teks_angka = st.text_area(
    "Histori angka (satu per baris):",
    height=250,
    value="""123456
789012
345678
901234
567890
234567
890123
456789
012345
678901
112233
445566
778899
135790
246801
987654
321098
654321
111222
333444
555666
777888
999000
121212
343434""",
    help="Pastikan semua angka 6 digit, tanpa spasi atau karakter lain."
)

# Parsing data
angka_list = [
    x.strip().zfill(6)
    for x in teks_angka.splitlines()
    if x.strip().isdigit() and len(x.strip()) == 6
]

if len(angka_list) < 15:
    st.warning(f"⚠️ Masukkan minimal **15 angka** histori agar model bisa belajar.")
    st.stop()

st.success(f"✅ Data berhasil dimuat: {len(angka_list)} angka 6-digit.")

# ======== MODEL CHOICE =========
st.subheader("🧠 Pilih Model Prediksi")
model_choice = st.selectbox(
    "Pilih algoritma prediksi:",
    ["Markov Chain", "LSTM Digit", "GRU Digit"],
    help="Markov Chain: berdasarkan pola transisi. LSTM/GRU: berdasarkan pola panjang deret waktu."
)

# Input angka terakhir
input_angka = st.text_input(
    "Angka terakhir (untuk prediksi):",
    value=angka_list[-1] if angka_list else "123456",
    max_chars=6,
    help="Masukkan angka terakhir dari histori Anda. Harus 6 digit."
)

if len(input_angka) != 6 or not input_angka.isdigit():
    st.error("❌ Angka terakhir harus 6 digit numerik!")
    st.stop()

# ======== HELPER FUNCTIONS =========

def angka_to_digit_array(data):
    """Konversi list string angka ke array numpy 2D (n, 6)"""
    return np.array([[int(d) for d in list(a)] for a in data], dtype=np.float32)

def build_markov_model(data):
    """Bangun transition matrix Markov Chain"""
    transition = defaultdict(list)
    for i in range(len(data) - 1):
        transition[data[i]].append(data[i + 1])
    return transition

def prediksi_markov(current, transition, n=5):
    """Prediksi n angka berikutnya menggunakan Markov Chain"""
    candidates = transition.get(current, [])
    if not candidates:
        # Jika tidak ada transisi, generate acak
        candidates = [str(random.randint(0, 999999)).zfill(6) for _ in range(100)]
    return random.choices(candidates, k=n)

def train_lstm_gru_model(data, use_gru=False, sequence_length=5):
    """Train LSTM atau GRU model untuk prediksi digit (6 digit)"""
    X = angka_to_digit_array(data[:-1])
    y = angka_to_digit_array(data[1:])
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.transform(y)
    
    # Buat timeseries generator
    gen = TimeseriesGenerator(X_scaled, y_scaled, length=sequence_length, batch_size=1)
    
    if len(gen) == 0:
        raise ValueError(f"Tidak cukup data untuk membuat sequence dengan panjang {sequence_length}. Masukkan lebih banyak data.")
    
    model = Sequential([
        (GRU if use_gru else LSTM)(64, activation='relu', input_shape=(sequence_length, 6)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(6)  # Output: 6 digit
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Latih model
    with st.spinner("⏳ Melatih model... (ini bisa memakan waktu 10-40 detik)"):
        history = model.fit(gen, epochs=20, verbose=0)
    
    return model, scaler, history

def prediksi_multi(model, scaler, last_sequence, n=5, sequence_length=5):
    """Prediksi n angka berikutnya menggunakan model LSTM/GRU"""
    out = []
    current_seq = last_sequence.copy()
    
    for _ in range(n):
        # Prediksi 1 langkah
        pred_scaled = model.predict(np.expand_dims(current_seq, axis=0), verbose=0)
        pred_raw = scaler.inverse_transform(pred_scaled)
        
        # Konversi ke angka 0-9
        pred_digits = np.clip(np.round(pred_raw).astype(int), 0, 9)
        pred_str = ''.join(map(str, pred_digits.flatten()))
        out.append(pred_str)
        
        # Update sequence untuk prediksi selanjutnya
        new_row = pred_digits.flatten()
        current_seq = np.vstack([current_seq[1:], new_row]) if len(current_seq) > 1 else np.array([new_row])
        
        # Pastikan shape tetap (sequence_length, 6)
        if len(current_seq) < sequence_length:
            pad = np.zeros((sequence_length - len(current_seq), 6))
            current_seq = np.vstack([pad, current_seq])
    
    return out

def hitung_akurasi(data, model_type="Markov", sequence_length=5, test_size=10):
    """Hitung akurasi top-1, top-3, top-5 pada subset data uji"""
    if len(data) < test_size + sequence_length + 1:
        raise ValueError(f"Data tidak cukup. Butuh minimal {test_size + sequence_length + 1} angka.")
    
    benar = {'top1': 0, 'top3': 0, 'top5': 0}
    total = 0
    
    # Uji dari index test_size sampai akhir
    start_idx = len(data) - test_size - 1
    end_idx = len(data) - 1
    
    for i in range(start_idx, end_idx):
        try:
            input_val = data[i]
            target = data[i + 1]
            
            if model_type == "Markov":
                transition = build_markov_model(data[:i+1])
                pred_list = prediksi_markov(input_val, transition, n=5)
            else:
                # LSTM/GRU
                model, scaler, _ = train_lstm_gru_model(data[:i+1], use_gru=(model_type=="GRU"), sequence_length=sequence_length)
                X = angka_to_digit_array(data[:i+1])
                scaler.fit(X)
                X_scaled = scaler.transform(X)
                last_seq = X_scaled[-sequence_length:]
                pred_list = prediksi_multi(model, scaler, last_seq, n=5, sequence_length=sequence_length)
            
            if target == pred_list[0]:
                benar['top1'] += 1
            if target in pred_list[:3]:
                benar['top3'] += 1
            if target in pred_list:
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
st.subheader("🎯 Prediksi Angka Berikutnya")

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
            last_seq = X_scaled[-5:]  # Ambil 5 angka terakhir
            prediksi = prediksi_multi(model, scaler, last_seq, n=5, sequence_length=5)
        
        st.success(f"✨ **Prediksi 5 angka berikutnya:**")
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

# Plot histori 5 angka terakhir
hist_data = angka_list[-5:]
hist_digits = [[int(d) for d in a] for a in hist_data]
for digit_pos in range(6):
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

# Plot prediksi
pred_digits = [[int(d) for d in p] for p in prediksi]
for digit_pos in range(6):
    fig.add_trace(
        go.Scatter(
            x=[f"Pred-{i}" for i in range(1,6)],
            y=[p[digit_pos] for p in pred_digits],
            mode='lines+markers',
            name=f'Posisi {digit_pos+1} (Prediksi)',
            line=dict(color=f'rgba(231, 76, 60, 0.7)', dash='dash'),
            showlegend=True
        ),
        row=1, col=1
    )

fig.update_layout(
    title="Perbandingan Digit Histori vs Prediksi (6 Digit)",
    xaxis_title="Langkah",
    yaxis_title="Nilai Digit (0-9)",
    height=400,
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

# ======== AKURASI =========
st.subheader("🔍 Uji Akurasi Model")

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
            
            st.markdown("### 📊 Hasil Akurasi:")
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
                title="Akurasi Prediksi (Top-1, Top-3, Top-5)",
                yaxis_title="Persentase (%)",
                yaxis_range=[0, 100],
                height=300,
                template="plotly_white"
            )
            st.plotly_chart(fig_acc, use_container_width=True)
            
            # Interpretasi
            if acc['top5'] > 20:
                st.info("💡 **Interpretasi**: Model menunjukkan pola tertentu dalam data — mungkin ada bias atau tren kecil. Tapi tetap tidak bisa dipercaya untuk prediksi nyata.")
            else:
                st.warning("⚠️ **Interpretasi**: Model tidak menemukan pola kuat. Ini sesuai ekspektasi — togel bersifat acak!")
                
        except Exception as e:
            st.error(f"❌ Gagal menghitung akurasi: {str(e)}")

# ======== INFO TEKNIS =========
with st.expander("📚 Penjelasan Teknis & Cara Kerja"):
    st.markdown("""
    ### 🔍 Bagaimana Model Ini Bekerja?

    #### **1. Markov Chain**
    - Membangun daftar “transisi” antar angka 6-digit.
    - Jika `123456` sering diikuti oleh `789012`, maka saat masuk `123456`, probabilitas `789012` naik.
    - **Kelebihan**: Cepat, sederhana, tidak butuh banyak data.
    - **Kekurangan**: Hanya lihat satu langkah sebelumnya — tidak paham pola panjang.

    #### **2. LSTM / GRU**
    - Menggunakan jaringan saraf rekursif untuk melihat **5 angka terakhir sebagai urutan waktu**.
    - Belajar pola kompleks: misalnya, jika digit pertama cenderung naik setiap 3 langkah.
    - **Kelebihan**: Bisa tangkap pola jangka panjang.
    - **Kekurangan**: Butuh banyak data, mudah overfit, sangat lambat.

    ### ⚠️ Penting!
    - **Togel 6 digit itu acak murni** — setiap angka punya peluang 1/1.000.000.
    - Model ini hanya meniru pola statistik dalam data yang kamu berikan — bukan memprediksi angka berikutnya yang "benar".
    - Aplikasi ini **hanya untuk pembelajaran** konsep AI dan deret waktu.
    """)
    
    st.markdown("#### 📈 Catatan Performa")
    st.markdown("""
    - Dalam data acak, akurasi top-5 biasanya **< 5%**.
    - Jika akurasi > 10%, itu karena data kamu **tidak acak** (misal: pola manusia, tanggal lahir, nomor HP).
    - Jika akurasi > 20%, kemungkinan besar data kamu **dibuat-buat** atau memiliki **bias kuat**.
    """)

# ======== FOOTER =========
st.divider()
st.caption("""
📌 **Disclaimer**: Aplikasi ini dibuat untuk tujuan edukasi dan eksperimen ilmu data.  
**Tidak direkomendasikan untuk digunakan dalam perjudian nyata.**  
Prediksi togel tidak mungkin akurat karena sifatnya acak. Gunakan pengetahuan ini untuk belajar AI, bukan untuk menang.
""")
