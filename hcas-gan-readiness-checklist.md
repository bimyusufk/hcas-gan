# HCAS-GAN Readiness Checklist & To-Do Implementasi

Dokumen ini dipakai untuk menilai **kesiapan implementasi** HCAS-GAN secara objektif, lalu menerjemahkannya ke **to-do yang mudah dipahami** oleh tim.

---

## Cara Pakai Singkat

1. Tandai setiap item dengan status:
   - `[ ]` Belum
   - `[~]` Sedang dikerjakan
   - `[x]` Selesai
2. Hitung skor readiness:
   - **Siap implementasi R&D**: minimal **14/24** item selesai, dengan mayoritas item **Kritis** sudah terpenuhi.
   - **Siap scale-up/production pilot**: minimal **20/24** item selesai, dan semua item kritis wajib selesai.

---

## A. Checklist Kritis (Wajib Selesai Dulu)

### 1) Desain Objective & Gradient Flow
- [x] Definisi objective final Generator dan Discriminator ditulis jelas (rumus + deskripsi).
- [x] Dipastikan **saliency loss benar-benar mengalir gradien ke Generator**.
- [x] Bobot model saliency (DeepGaze) di-freeze, tetapi gradien terhadap input tetap tersedia.
- [x] Ada uji kecil (sanity check) yang membuktikan update bobot Generator berubah saat saliency loss aktif.

### 2) Data Contract & Kualitas Data
- [ ] Struktur data final disepakati (`raw_environments/`, `masks/`, `processed/`).
- [x] Format mask baku ditetapkan (nilai, channel, ukuran, validasi non-empty).
- [x] Split data `train/val/test` terdokumentasi dan reproducible.
- [x] Pipeline preprocessing deterministik (seed, resize, normalisasi) sudah ditulis.

### 3) Stabilitas Training GAN
- [x] Aturan update G:D disepakati (misal 1:1 atau 1:2) dan terdokumentasi.
- [~] Mekanisme anti-instability dipilih (contoh: label smoothing / regularization / spectral norm).
- [x] Checkpointing + resume training berjalan dan sudah diuji.
- [ ] Kriteria early stopping / model selection ditetapkan.

### 4) Evaluasi Objektif Anti-Saliency
- [x] Metrik tekstur (SSIM, LPIPS) tersedia dan dapat dieksekusi otomatis.
- [x] Metrik saliency khusus ditetapkan (contoh: rata-rata atensi di area mask, delta vs baseline).
- [ ] Ada evaluasi per-skenario (hutan, gurun, urban) agar hasil tidak bias domain.
- [ ] Format laporan eksperimen standar disepakati (tabel metrik + visual sample).

---

## B. Checklist Pendukung (Meningkatkan Kematangan)

### 5) Reproducibility & MLOps
- [x] Konfigurasi terpusat (`config.yaml`) dipakai konsisten di seluruh script.
- [~] Logging eksperimen (mis. W&B) aktif: loss, grad norms, contoh output berkala.
- [~] Semua dependency terkunci (`requirements.txt`/lock file) dan environment bisa direplikasi.
- [x] Seed global (Python/NumPy/PyTorch/CUDA) di-set dan diverifikasi.

### 6) Engineering Readiness
- [x] Struktur modul `src/models`, `src/data`, `src/training`, `src/utils` sudah dipakai konsisten.
- [x] Ada test dasar (minimal smoke test data loader + forward pass + 1 step training).
- [ ] README runbook internal tersedia (cara setup, train, evaluate, troubleshoot).
- [ ] Risiko hardware terkelola (fallback batch size, gradient accumulation, OOM handling).

---

## Ringkasan Skor

- Total item: **24**
- Item selesai: **15**
- Item parsial: **3**
- Item belum: **6**
- Persentase readiness (selesai): **62.5%**

**Keputusan saat ini:**
- [ ] Belum siap implementasi
- [x] Siap implementasi R&D
- [ ] Siap pilot produksi

---

## To-Do Implementasi (Mudah Dipahami)

> Fokus: dari "dokumen bagus" jadi "pipeline yang benar-benar jalan".

### Minggu 1 — Fondasi Teknis
- [ ] Finalkan rumus loss di dokumen teknis (versi final, tidak ambigu).
- [ ] Implement `src/training/loss.py` + unit sanity check gradien.
- [ ] Implement `src/data/dataset_loader.py` + validasi mask non-empty.
- [ ] Buat script smoke test 1 batch end-to-end.

### Minggu 2 — Training Dasar Stabil
- [ ] Implement `generator.py` dan `discriminator.py` baseline.
- [ ] Buat `trainer.py` yang bisa jalan minimal 1 epoch tanpa error.
- [ ] Aktifkan logging (loss G/D, sample image, waktu per iterasi).
- [ ] Simpan checkpoint berkala + bisa resume dari checkpoint.

### Minggu 3 — Integrasi Saliency & Evaluasi
- [ ] Integrasikan wrapper DeepGaze dengan mode freeze bobot.
- [ ] Pastikan saliency loss aktif mempengaruhi update Generator.
- [ ] Tambah evaluasi SSIM/LPIPS + metrik anti-saliency.
- [ ] Jalankan eksperimen awal 3 environment (small subset).

### Minggu 4 — Hardening & Keputusan Go/No-Go
- [ ] Tuning hyperparameter minimal 2–3 kombinasi.
- [ ] Lakukan ablation (dengan vs tanpa saliency loss).
- [ ] Susun laporan hasil + contoh visual komparatif.
- [ ] Putuskan: lanjut scale-up atau revisi arsitektur.

---

## Definisi Done (Agar Tidak Abu-Abu)

Sebuah task dianggap **Done** jika memenuhi semua:
1. Kodenya berjalan pada environment yang sama.
2. Ada bukti hasil (log/metric/artifact).
3. Tidak ada error kritis saat smoke test.
4. Perubahan tercatat jelas di commit/notes.

---

## Catatan Risiko Utama

- **Risiko 1: Saliency loss tidak efektif** → Mitigasi: sanity check gradien per modul.
- **Risiko 2: VRAM tidak cukup** → Mitigasi: kecilkan batch, aktifkan gradient accumulation.
- **Risiko 3: Mode collapse GAN** → Mitigasi: monitor loss pattern + visual sample tiap interval.
- **Risiko 4: Evaluasi bias domain** → Mitigasi: uji lintas environment, bukan satu scene saja.
