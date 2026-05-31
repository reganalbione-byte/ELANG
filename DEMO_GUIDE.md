# Panduan Demo & Screenshot ELANG (untuk Proposal)

Langkah per halaman: **apa yang diinput, apa yang diklik, dan output apa yang
di-screenshot**. Tiap bagian punya saran caption untuk proposal.

## Persiapan (sekali saja)

```powershell
.venv\Scripts\activate
streamlit run app.py
```

File demo sudah disiapkan & sudah diuji menghasilkan output — semuanya ada di `data/`:

| File | Dipakai di tab | Hasil yang sudah terverifikasi |
|------|----------------|-------------------------------|
| `sample_plate.jpg` | Image (ANPR) | Mobil terdeteksi, plat terbaca **`B 2761 SUA`** (valid, DKI Jakarta) |
| `sample_traffic.jpg` | Image (deteksi) | 6 kendaraan terdeteksi (scene ramai) |
| `sample_video.mp4` | Video | 60 dtk, sampai **20 kendaraan**/frame |
| `sample_video_short.mp4` | Video | 24 dtk, ~12 kendaraan (alternatif lebih cepat) |
| `sample_violations.csv` | Heatmap | 25 baris, 5 koridor Jakarta (hotspot jelas) |
| `sample.jpg` | Image | Foto bus bawaan (deteksi sederhana) |

> Tip screenshot: lebarkan window browser, dan di sidebar Streamlit aktifkan
> **Settings → Wide mode** kalau perlu. Untuk Windows: `Win + Shift + S` (Snipping Tool).

---

## Tab 1 — 📷 Image: Deteksi Kendaraan

**Input:** upload `data/sample_traffic.jpg` (jangan centang ANPR dulu).

**Output untuk di-screenshot:**
- Gambar beranotasi dengan **kotak hijau + label** tiap kendaraan.
- Panel kanan: metric **Total vehicles**, **Avg confidence**, dan chart **By class**.

**Caption proposal:** *"Deteksi kendaraan multi-kelas (mobil/motor/bus/truk)
berbasis YOLOv8 pada satu frame CCTV."*

---

## Tab 1b — 📷 Image: ANPR (Pembacaan Plat)

**Input:** upload `data/sample_plate.jpg`.
**Klik:**
1. Centang **"Run ANPR on each detected vehicle"**.
2. **Enhance mode** = `auto`.
3. (Opsional) centang **"Show preprocessing preview"** untuk screenshot before/after.

**Output untuk di-screenshot (3 hal terkuat):**
- Gambar dengan plat **`B 2761 SUA`** tertulis kuning di atas kendaraan + tabel **Plates read**.
- Blok **Plate validation**: badge hijau *"🟢 Valid: B 2761 SUA — region B (DKI Jakarta), type=reguler"*.
- **ANPR Accuracy Report**: metric crops/plates/valid + progress bar + catatan target akurasi jujur (85% / 65%).

**Caption proposal:** *"ANPR membaca & memvalidasi plat terhadap format TNKB
Indonesia (23 kode wilayah, 4 tipe plat), lengkap dengan laporan akurasi."*

---

## Tab 2 — 🎞️ Video: Tracking + Zona Pelanggaran

**Sidebar (atur dulu):** `Max video frames` = **150**, `Frame stride` = **3**.

**Di tab Video:**
1. **Input source** = `Upload File` → upload `data/sample_video.mp4`.
2. Centang **Enable DeepSORT tracking**.
3. **Min frames in zone → violation** = **3**.
4. Di kotak **Restricted-zone polygon**, tempel (koordinat piksel, frame 640×360):
   ```
   180,150
   470,150
   470,310
   180,310
   ```

**Output untuk di-screenshot:**
- Preview frame: kotak dengan **#ID track + label**, dan **overlay zona merah** transparan.
- Tabel **Tracks** (track_id, label, duration_frames, frames_in_zone).
- Metric **Violators (in-zone ≥ threshold)**.
- Chart **Total detections across frames** + **Per-frame timeline**.

**Caption proposal:** *"Tracking multi-objek (DeepSORT) menjaga ID antar-frame dan
menghitung durasi kendaraan di zona terlarang untuk menandai pelanggaran."*

> Catatan: di highway kendaraan lewat cepat, makanya threshold dibuat kecil (3).
> Untuk demo "parkir liar" yang menahan lama, pakai clip kendaraan diam/parkir.

---

## Tab 3 — 🗺️ Heatmap + Optimizer + Export E-TLE

**Input:** buka tab → hapus isi CSV bawaan → tempel isi `data/sample_violations.csv`
(atau klik **"…or upload a CSV"** lalu pilih file itu).

**Output untuk di-screenshot:**
1. **Peta Folium**: heatmap + lingkaran merah hotspot di koridor Jakarta.
2. Tabel **Top hotspots** (lat, lon, count).
3. **Officer placement optimizer**: tabel kandidat + skor + alasan
   (biarkan kandidat default, atau tempel beberapa `lat,lon`).
4. Scroll ke **📤 Export ke E-TLE** → klik **Generate Mock Violations** (mis. 10) →
   screenshot tabel violations + tombol **Download JSON / CSV** + badge
   *"Format-compatible dengan standar E-TLE POLRI"*.

**Caption proposal:** *"Agregasi spatial-temporal pelanggaran → hotspot →
rekomendasi penempatan petugas/kamera, dan ekspor ke envelope standar E-TLE POLRI."*

---

## Tab 4 — 💬 CRM Classifier

**Mode Single report:**
- Ketik: `Ada motor parkir di atas trotoar depan Indomaret Sudirman` → klik **Classify**.
- Screenshot: **Kategori** (`parkir_liar`/`trotoar_dipakai_kendaraan`), **Confidence**, badge **Urgency**.
- Coba juga: `Kecelakaan parah motor vs truk di Cawang, ada korban luka` → badge **🔴 Urgency: HIGH**.

**Mode Batch (CSV):**
- Centang **"Atau pakai contoh inline (5 laporan)"**.
- Screenshot: tabel hasil (text/category/confidence/urgency), **pie chart distribusi kategori**,
  dan metric **Low / Medium / High**.

**Caption proposal:** *"Klasifikasi laporan warga (Sentence-BERT multilingual +
Logistic Regression) ke 6 kategori pelanggaran + level urgensi otomatis."*

---

## ⚠️ Catatan lisensi (penting sebelum publikasi)

Gambar & video demo diunduh dari sumber publik (Wikimedia Commons / YouTube) **untuk
keperluan uji prototipe**. Untuk proposal yang dipublikasikan/diserahkan ke juri,
sebaiknya:
- Ganti dengan footage CCTV/foto milik sendiri atau yang berlisensi jelas, **atau**
- Cantumkan atribusi sumber di caption gambar.

`sample_plate.jpg` & `sample_traffic.jpg` berasal dari Wikimedia Commons (kategori
taksi/plat Indonesia, umumnya CC BY-SA). `sample_video*.mp4` dari hasil pencarian
YouTube. Verifikasi lisensi tiap aset sebelum dipakai di dokumen final.
