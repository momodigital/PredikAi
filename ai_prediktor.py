#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 AI PREDICTOR 6 ANGKA UNIK - DENGAN COPY PASTE MASSAL
- Bisa copy paste puluhan data sekaligus
- Otomatis membaca semua angka 4 digit
- Support 20, 30, bahkan 100 putaran
"""

import numpy as np
from collections import defaultdict, Counter
import math
import os
import sys
import re

# ======== KELAS AI PREDICTOR =========

class AIPredictor6Angka:
    """
    AI untuk memprediksi 6 angka unik berdasarkan
    analisis multi-metode dari data historis
    """
    
    def __init__(self, data_histori):
        """Inisialisasi AI dengan data histori"""
        self.data_histori = data_histori
        self.frekuensi_digit = Counter()
        self.frekuensi_2d = Counter()
        self.posisi = {0: Counter(), 1: Counter(), 2: Counter(), 3: Counter()}
        self.transisi_2d = defaultdict(list)
        self.korelasi = np.zeros((10, 10))
        
        # Hitung semua statistik dari data histori
        self._hitung_statistik_dasar()
        print(f"\n✅ AI siap! Data histori: {len(data_histori)} putaran")
    
    def _hitung_statistik_dasar(self):
        """Menghitung semua statistik dari data histori"""
        for angka in self.data_histori:
            # Frekuensi digit
            for digit in angka:
                self.frekuensi_digit[int(digit)] += 1
            
            # 2D (2 digit terakhir)
            dua_d = angka[-2:]
            self.frekuensi_2d[dua_d] += 1
            
            # Posisi
            for pos, digit in enumerate(angka):
                self.posisi[pos][int(digit)] += 1
        
        # Transisi Markov Chain untuk 2D
        for i in range(len(self.data_histori) - 1):
            state_sekarang = self.data_histori[i][-2:]
            state_berikut = self.data_histori[i+1][-2:]
            self.transisi_2d[state_sekarang].append(state_berikut)
        
        # Korelasi antar digit
        for angka in self.data_histori:
            digits = [int(d) for d in angka]
            for i in range(4):
                for j in range(i+1, 4):
                    self.korelasi[digits[i]][digits[j]] += 1
                    self.korelasi[digits[j]][digits[i]] += 1
    
    # ======== 8 METODE AI =========
    
    def _metode_frekuensi_dasar(self, digit):
        """METODE 1: Frekuensi kemunculan (Bobot 20%)"""
        max_freq = max(self.frekuensi_digit.values()) if self.frekuensi_digit else 1
        freq = self.frekuensi_digit.get(digit, 0)
        return (freq / max_freq) * 20
    
    def _metode_frekuensi_2d(self, digit):
        """METODE 2: Frekuensi di 2D (Bobot 25%)"""
        # Hitung berapa kali digit ini muncul di posisi 2D
        freq_2d = 0
        for dua_d, count in self.frekuensi_2d.items():
            if str(digit) in dua_d:
                freq_2d += count
        
        max_freq_2d = max(self.frekuensi_2d.values()) * 2 if self.frekuensi_2d else 1
        return (freq_2d / max_freq_2d) * 25 if max_freq_2d > 0 else 0
    
    def _metode_posisi(self, digit, angka_4digit):
        """METODE 3: Analisis posisi (Bobot 15%)"""
        skor = 0
        # Fokus ke posisi 3 dan 4 (karena untuk 2D)
        for pos in [2, 3]:
            max_pos = max(self.posisi[pos].values()) if self.posisi[pos] else 1
            freq_pos = self.posisi[pos].get(digit, 0)
            skor += (freq_pos / max_pos) * 7.5
        return skor
    
    def _metode_markov(self, digit):
        """METODE 4: Markov Chain (Bobot 15%)"""
        if not self.data_histori:
            return 0
        
        last_2d = self.data_histori[-1][-2:]
        skor = 0
        
        if last_2d in self.transisi_2d:
            next_states = self.transisi_2d[last_2d]
            for dua_d in next_states:
                if str(digit) in dua_d:
                    skor += 2
        return min(skor, 15)
    
    def _metode_korelasi(self, digit, angka_4digit):
        """METODE 5: Korelasi dengan digit terakhir (Bobot 10%)"""
        if not angka_4digit:
            return 0
        
        last_digits = [int(d) for d in angka_4digit]
        skor = 0
        for d in last_digits:
            skor += self.korelasi[d][digit]
        return min(skor / 5, 10)
    
    def _metode_statistik(self, digit):
        """METODE 6: Statistik lanjutan (Bobot 5%)"""
        skor = 0
        total_digit = len(self.data_histori) * 4
        rata_rata = total_digit / 10
        
        freq = self.frekuensi_digit.get(digit, 0)
        
        # Digit yang jarang muncul dapat bonus (anti-apatis)
        if freq < rata_rata:
            skor += (rata_rata - freq) * 0.5
        
        # Digit yang sering muncul dapat sedikit penalti
        if freq > rata_rata * 1.5:
            skor -= (freq - rata_rata * 1.5) * 0.3
        
        return max(0, min(skor, 5))
    
    def _metode_pola_2d(self, digit):
        """METODE 7: Pola 2D berulang (Bobot 5%)"""
        skor = 0
        
        # Cari 2D yang sering muncul
        for dua_d, count in self.frekuensi_2d.most_common(5):
            if count >= 2:  # 2D yang muncul lebih dari sekali
                if str(digit) in dua_d:
                    skor += 1
        
        return min(skor, 5)
    
    def _metode_antisipasi(self, digit):
        """METODE 8: Antisipasi digit jarang muncul (Bobot 5%)"""
        freq = self.frekuensi_digit.get(digit, 0)
        total_putaran = len(self.data_histori)
        
        # Jika digit sangat jarang relatif terhadap putaran
        if freq <= total_putaran * 0.2:  # Kurang dari 20% putaran
            return 5  # Bonus penuh
        elif freq <= total_putaran * 0.3:  # Kurang dari 30% putaran
            return 3  # Bonus sedang
        else:
            return 0
    
    def prediksi_6_angka(self, angka_4digit, tampilkan_detail=False):
        """
        PREDIKSI UTAMA: Menghasilkan 6 angka unik berdasarkan
        kombinasi semua metode AI
        """
        
        skor = {}
        
        if tampilkan_detail:
            print("\n🧠 AI MEMPROSES DENGAN 8 METODE...")
            print("-"*50)
        
        for digit in range(10):
            if tampilkan_detail:
                print(f"\n📊 Menghitung Digit {digit}:")
            
            # Kumpulkan skor dari semua metode
            m1 = self._metode_frekuensi_dasar(digit)
            m2 = self._metode_frekuensi_2d(digit)
            m3 = self._metode_posisi(digit, angka_4digit)
            m4 = self._metode_markov(digit)
            m5 = self._metode_korelasi(digit, angka_4digit)
            m6 = self._metode_statistik(digit)
            m7 = self._metode_pola_2d(digit)
            m8 = self._metode_antisipasi(digit)
            
            if tampilkan_detail:
                print(f"   M1(Frekuensi): {m1:.1f}")
                print(f"   M2(Frekuensi 2D): {m2:.1f}")
                print(f"   M3(Posisi): {m3:.1f}")
                print(f"   M4(Markov): {m4:.1f}")
                print(f"   M5(Korelasi): {m5:.1f}")
                print(f"   M6(Statistik): {m6:.1f}")
                print(f"   M7(Pola 2D): {m7:.1f}")
                print(f"   M8(Antisipasi): {m8:.1f}")
            
            # Total skor
            total = m1 + m2 + m3 + m4 + m5 + m6 + m7 + m8
            skor[digit] = total
            
            if tampilkan_detail:
                print(f"   ➡ TOTAL: {total:.1f}")
        
        # Urutkan berdasarkan skor tertinggi
        sorted_digits = sorted(skor.items(), key=lambda x: x[1], reverse=True)
        
        # Ambil 6 digit teratas
        self.angka_terpilih = [digit for digit, _ in sorted_digits[:6]]
        
        return self.angka_terpilih, skor, sorted_digits


# ======== FUNGSI INPUT DATA HISTORI MASSAL =========

def input_histori_massal():
    """
    Fungsi untuk input data histori dengan COPY PASTE
    Bisa paste puluhan angka sekaligus
    """
    print("\n" + "="*80)
    print("📥 INPUT DATA HISTORI 4 DIGIT - COPY PASTE MASSAL")
    print("="*80)
    print("""
    CARA PENGGUNAAN:
    1. Copy semua angka 4 digit dari sumber Anda
    2. Paste di sini (bisa langsung banyak baris)
    3. Program akan otomatis membaca semua angka 4 digit
    4. Ketik 'selesai' pada baris baru setelah paste
    
    CONTOH FORMAT:
    0141
    0946
    9928
    8140
    7657
    (dan seterusnya)
    """)
    
    print("\n📋 Silakan paste data Anda (minimal 15 putaran):")
    print("-"*50)
    
    data = []
    baris_ke = 1
    
    while True:
        try:
            baris = input()
            
            # Cek perintah selesai
            if baris.lower() in ['selesai', 'exit', 'q', '']:
                break
            
            # Cari semua angka 4 digit dalam baris (pisahkan dengan spasi, koma, dll)
            angka_dalam_baris = re.findall(r'\b\d{4}\b', baris)
            
            for angka in angka_dalam_baris:
                if angka.isdigit() and len(angka) == 4:
                    data.append(angka)
                    print(f"  ✅ {angka}", end="")
                    if len(data) % 5 == 0:
                        print()
                    else:
                        print("  ", end="")
            
            baris_ke += 1
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Input dibatalkan")
            return None
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue
    
    print()  # New line
    
    if len(data) < 15:
        print(f"\n⚠️ Data kurang dari 15 putaran! (Hanya {len(data)} putaran)")
        print("   Tetap lanjut? Hasil mungkin kurang akurat.")
        lanjut = input("   Lanjutkan? (y/n): ").strip().lower()
        if lanjut != 'y':
            return None
    
    return data


def input_manual():
    """
    Alternatif input manual satu per satu
    """
    print("\n" + "="*80)
    print("📥 INPUT DATA HISTORI 4 DIGIT - MANUAL")
    print("="*80)
    print("Masukkan angka 4-digit satu per baris")
    print("Ketik 'selesai' jika sudah\n")
    
    data = []
    while True:
        try:
            angka = input(f"Putaran ke-{len(data)+1}: ").strip()
            
            if angka.lower() in ['selesai', 'exit', 'q']:
                if len(data) < 15:
                    print(f"⚠️ Minimal 15 data! Saat ini: {len(data)}")
                    continue
                break
            
            if angka.isdigit() and len(angka) == 4:
                data.append(angka)
                print(f"✅ {angka} diterima")
            else:
                print("❌ Harus 4 digit angka!")
                
        except KeyboardInterrupt:
            print("\n\n⚠️ Input dibatalkan")
            return None
    
    return data


# ======== FUNGSI MENU UTAMA =========

def clear_screen():
    """Bersihkan layar terminal"""
    os.system('clear' if os.name == 'posix' else 'cls')


def main():
    """Program utama"""
    
    clear_screen()
    
    print("="*90)
    print("  🎯 AI PREDICTOR 6 ANGKA UNIK - DENGAN COPY PASTE MASSAL")
    print("="*90)
    print("""
    FITUR:
    • Bisa COPY PASTE puluhan angka sekaligus
    • Support 20, 30, bahkan 100+ putaran
    • Otomatis membaca semua angka 4 digit
    • AI dengan 8 metode perhitungan
    • Hasil 6 angka unik untuk setiap pasaran
    """)
    
    # ===== PILIH METODE INPUT =====
    print("\n📌 PILIH METODE INPUT HISTORI:")
    print("   1. COPY PASTE MASSAL (cepat untuk banyak data)")
    print("   2. Input manual satu per satu")
    
    pilihan = input("\nPilih (1/2): ").strip()
    
    if pilihan == '1':
        histori = input_histori_massal()
    else:
        histori = input_manual()
    
    if not histori:
        print("\n❌ Tidak ada data histori, program berhenti.")
        return
    
    # ===== TAMPILKAN RINGKASAN =====
    clear_screen()
    print("\n" + "="*90)
    print("📊 RINGKASAN DATA HISTORI")
    print("="*90)
    print(f"\n✅ Total putaran: {len(histori)} data")
    
    # Tampilkan 10 data terakhir
    print("\n📋 10 DATA TERAKHIR:")
    for i, h in enumerate(histori[-10:], max(1, len(histori)-9)):
        print(f"  {i:3}. {h}")
    
    # Statistik cepat
    digit_counter = Counter()
    for angka in histori:
        for digit in angka:
            digit_counter[int(digit)] += 1
    
    print("\n📊 FREKUENSI DIGIT:")
    for digit in range(10):
        freq = digit_counter.get(digit, 0)
        persen = (freq / (len(histori) * 4)) * 100
        bar = "█" * int(persen)
        print(f"  Digit {digit}: {freq:2}x ({persen:4.1f}%) {bar}")
    
    # ===== INISIALISASI AI =====
    print("\n⚙️  Menginisialisasi AI...")
    ai = AIPredictor6Angka(histori)
    
    input("\nTekan ENTER untuk mulai prediksi pasaran...")
    
    # ===== LOOP UNTUK BERBAGAI PASARAN =====
    while True:
        clear_screen()
        print("\n" + "="*90)
        print("  🎯 INPUT 4 DIGIT UNTUK PASARAN INI")
        print("="*90)
        print(f"📊 Data histori: {len(histori)} putaran")
        print(f"📌 4 digit terakhir: {histori[-1]}")
        
        # Input 4 digit untuk pasaran ini
        while True:
            angka_pasaran = input("\n📝 Masukkan 4 digit angka pasaran: ").strip()
            
            if angka_pasaran.lower() == 'exit':
                print("\n👋 Program selesai!")
                return
            
            if angka_pasaran.isdigit() and len(angka_pasaran) == 4:
                break
            else:
                print("❌ Harus 4 digit angka! (atau ketik 'exit' untuk keluar)")
        
        # Tanya detail
        detail = input("\n🔍 Tampilkan detail perhitungan? (y/n): ").strip().lower()
        tampilkan_detail = (detail == 'y')
        
        # AI memproses
        print("\n⏳ AI sedang menganalisis...")
        angka_terpilih, semua_skor, sorted_digits = ai.prediksi_6_angka(angka_pasaran, tampilkan_detail)
        
        # ===== TAMPILKAN HASIL =====
        print("\n" + "="*90)
        print("  🎯 HASIL PREDIKSI AI - 6 ANGKA UNIK")
        print("="*90)
        
        print(f"\n📌 4 DIGIT INPUT: {angka_pasaran}")
        
        print("\n📊 SKOR SETIAP DIGIT:")
        print("-"*70)
        print("  Rank | Digit | Skor  | Grafik                    | Rekomendasi")
        print("-"*70)
        
        for rank, (digit, skor) in enumerate(sorted_digits, 1):
            bar_length = int(skor / 2)
            bar = "█" * bar_length
            if rank <= 3:
                rekom = "🔥 PILIHAN UTAMA"
            elif rank <= 6:
                rekom = "✅ BISA DIPAKAI"
            else:
                rekom = "⚠️ PERTIMBANGKAN"
            
            print(f"  {rank:2}   |   {digit}   | {skor:5.1f} | {bar:35} | {rekom}")
        
        print("\n" + "="*90)
        print(f"✨ 6 ANGKA UNIK TERPILIH: {angka_terpilih}")
        print("="*90)
        
        # Hitung kombinasi 2D
        kombinasi_2d = []
        for d1 in angka_terpilih:
            for d2 in angka_terpilih:
                dua_d = f"{d1}{d2}"
                kombinasi_2d.append(dua_d)
        
        print(f"\n📋 Total {len(kombinasi_2d)} KOMBINASI 2D:")
        for i, dua_d in enumerate(kombinasi_2d, 1):
            print(f"{dua_d} ", end="")
            if i % 15 == 0:
                print()
        print()
        
        # Rekomendasi 6 digit
        print("\n📌 REKOMENDASI 6 DIGIT (prioritas):")
        print(f"  1. {angka_pasaran}{angka_terpilih[0]}{angka_terpilih[1]}  (2D dari 2 digit terkuat)")
        print(f"  2. {angka_pasaran}{angka_terpilih[0]}{angka_terpilih[2]}  (2D dari digit 1 & 3)")
        print(f"  3. {angka_pasaran}{angka_terpilih[1]}{angka_terpilih[0]}  (2D dari digit 2 & 1)")
        print(f"  4. {angka_pasaran}{angka_terpilih[0]}{angka_terpilih[0]}  (2D double digit terkuat)")
        
        # Hitung persentase kepercayaan
        total_skor = sum(skor for _, skor in sorted_digits[:3])
        confidence = (total_skor / (sorted_digits[0][1] * 3)) * 100
        
        print(f"\n📈 TINGKAT KEPERCAYAAN AI: {confidence:.1f}%")
        print(f"    (Berdasarkan konsistensi {len(histori)} putaran data)")
        
        # ===== TANYA UNTUK PASARAN LAIN =====
        print("\n" + "="*90)
        lagi = input("\n❓ Hitung pasaran lain? (y/n): ").strip().lower()
        
        if lagi != 'y':
            print("\n👋 Terima kasih! Program selesai.")
            print("   Ingat: Ini hanya untuk pembelajaran AI!")
            break


# ======== JALANKAN PROGRAM =========

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Program dihentikan user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
