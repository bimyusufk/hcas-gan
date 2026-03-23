# HCAS-GAN v2 — Roadmap Komprehensif Implementasi

Dokumen ini adalah rencana eksekusi bertahap untuk meningkatkan kualitas pola dari “tekstur blend umum” menjadi lebih mendekati **camouflage tipikal militer** tanpa mengorbankan objektif anti-saliency.

---

## 1) Tujuan v2

### Tujuan utama

1. Mempertahankan objective HCAS lama:
   - natural blending,
   - anti-saliency.
2. Menambahkan objective baru:
   - **military-camo likeness** (warna, struktur, skala pola).

### Definisi sukses v2

- Secara visual, pattern terlihat lebih “military-like” (blotchy, terbatas palet, multi-skala).
- `g_sal` tetap kompetitif (tidak regress besar vs baseline v1).
- Discriminator tetap stabil (tidak mode collapse/divergence).
- Evaluasi visual panel + metrik kuantitatif menunjukkan peningkatan konsisten pada run replikasi.

---

## 2) Strategi Inti yang Diadopsi

## 2.1 Style prior (prioritas tertinggi)
- Tambah dataset referensi style camo (woodland/desert/digital).
- Tambah **style/texture loss** (feature + Gram) terhadap style bank.

## 2.2 Palette-constrained objective
- Ekstrak palet target (mis. k-means 4–6 warna per domain).
- Penalti jarak histogram / palette assignment output terhadap target distribution.

## 2.3 Frequency / pattern-scale regularization
- Cocokkan distribusi frekuensi (FFT radial / Laplacian pyramid energy).
- Targetkan struktur multi-skala (blob besar + detail kecil).

## 2.4 Curriculum untuk lambda_sal
- `lambda_sal` tidak langsung tinggi.
- Naik bertahap agar model belajar style dulu, lalu makin anti-saliency.

## 2.5 Sampling curriculum berdasarkan area mask
- Oversample contoh dengan area mask menengah-besar.
- Hindari sinyal style terlalu lemah karena mask terlalu kecil.

## 2.6 Opsi lanjutan: 2-stage pipeline
- **Stage A**: generator tile camouflage military-like.
- **Stage B**: adapter/blender ke lingkungan + anti-saliency.

---

## 3) Arsitektur Target v2 (Incremental, bukan sekali lompat)

## 3.1 v2.1 (quick-win, minimal intrusif)
- Tambah loss:
  - `L_style`
  - `L_palette`
  - `L_freq`
- Total objective generator:

$$
L_G = L_{adv} + \lambda_{sal}(e) L_{sal} + \lambda_{style} L_{style} + \lambda_{pal} L_{palette} + \lambda_{freq} L_{freq}
$$

- Tambah scheduler `lambda_sal(e)`.
- Tambah sampler berbasis bucket area mask.

## 3.2 v2.2 (stabilisasi kualitas style)
- Tambah texture discriminator opsional (`D_tex`) yang belajar membedakan “military-camo-like patch” vs non-camo patch.
- Ablation apakah `D_tex` lebih efektif dari kombinasi loss saja.

## 3.3 v2.3 (major redesign)
- Implementasi 2-stage:
  - Tile Generator (style-first)
  - Context Adapter (blend + anti-saliency)

---

## 4) Workstream Teknis

## 4.1 Data Workstream
- Kurasi style bank camo:
  - woodland
  - tropical/jungle
  - arid/desert
  - digital camo
- Buat metadata domain style (opsional untuk conditioning).
- Bangun script ekstraksi palette per domain.

## 4.2 Modeling Workstream
- Implementasi style loss module.
- Implementasi palette loss module.
- Implementasi frequency loss module.
- Integrasi lambda curriculum.
- Integrasi mask-area curriculum sampler.

## 4.3 Training & Evaluation Workstream
- Konfigurasi baseline v1 re-run (seed fixed) sebagai pembanding.
- Rancang matriks ablation run.
- Standarisasi output visual panel + scoring rubric.
- Rekam metrik + artifact per run.

## 4.4 Tooling & Product Workstream
- Integrasi inference web app sebagai alat inspeksi cepat.
- Export artifact otomatis per checkpoint:
  - composite panel
  - flat pattern 1x1
  - stats mask area

---

## 5) Rencana Ablation (Eksperimen Bertahap)

## 5.1 Baseline
- **Run A (baseline v1)**: `L_adv + L_sal`

## 5.2 Quick-win additions
- **Run B**: baseline + `L_style`
- **Run C**: Run B + `L_palette`
- **Run D**: Run C + `L_freq`

## 5.3 Curriculum additions
- **Run E**: Run D + `lambda_sal` schedule
- **Run F**: Run E + mask-area curriculum sampler

## 5.4 Advanced addition
- **Run G**: Run F + texture discriminator (`D_tex`)

### Kriteria pemilihan kandidat terbaik
1. Visual military-likeness (rubric score)
2. Anti-saliency tetap baik (`g_sal`, val/test)
3. Stabilitas GAN (`d_total` bounded, tanpa collapse)
4. Generalisasi visual lintas latar

---

## 6) Rubric Penilaian Visual (Praktis)

Skor 1–5 per item:

1. **Palette military-like** (terbatas, harmonis domain)
2. **Blob structure** (bukan noise acak homogen)
3. **Multi-scale pattern** (macro + micro detail)
4. **Edge blending** terhadap environment
5. **Target detectability** (sulit terlihat pada jarak pandang normal)

Total skor visual per sampel: 5–25.

---

## 7) Risk Register + Mitigasi

1. **Over-regularization style** → hasil terlalu “template-like”
   - Mitigasi: tuning `lambda_style`, random style-bank mix.

2. **Palette loss terlalu kuat** → detail tekstur mati
   - Mitigasi: warmup lambda palette.

3. **Freq loss tidak stabil**
   - Mitigasi: normalisasi energi frekuensi + clipping.

4. **Trade-off anti-saliency vs style**
   - Mitigasi: curriculum `lambda_sal`, multi-objective sweep.

5. **Kompleksitas training meningkat**
   - Mitigasi: fase v2.1 dulu sebelum `D_tex`/2-stage.

---

## 8) Timeline Implementasi Bertahap (Checklist To-Do)

## Fase 0 — Persiapan eksperimen (1–2 hari)
- [ ] Freeze baseline config + seed + data split
- [ ] Definisikan protokol evaluasi (visual + metrik)
- [ ] Siapkan folder artifact standar per run
- [ ] Dokumentasikan naming convention run/checkpoint

## Fase 1 — Data style bank & palette (2–4 hari)
- [ ] Kumpulkan referensi pattern camo (per domain)
- [ ] Bersihkan & resize style images
- [ ] Buat script ekstraksi palette (k-means)
- [ ] Simpan statistik palette per domain (JSON/NPY)
- [ ] Tambah sanity-check untuk style dataset

## Fase 2 — Implementasi loss baru v2.1 (3–5 hari)
- [ ] Tambah `src/training/loss_style.py`
- [ ] Tambah `src/training/loss_palette.py`
- [ ] Tambah `src/training/loss_frequency.py`
- [ ] Integrasi ke `HCASLoss` (config-driven on/off)
- [ ] Tambah unit sanity script untuk tiap loss

## Fase 3 — Curriculum training (2–3 hari)
- [ ] Tambah scheduler `lambda_sal(epoch)`
- [ ] Tambah mask-area bucket sampler (small/medium/large)
- [ ] Tambah logging TensorBoard untuk setiap komponen loss
- [ ] Verifikasi stabilitas 50–100 epoch smoke run

## Fase 4 — Ablation quick-win (3–6 hari)
- [ ] Jalankan Run A–F sesuai matriks ablation
- [ ] Simpan visual panel per checkpoint interval
- [ ] Rekap metrik train/val/test dalam tabel komparasi
- [ ] Pilih 2 kandidat terbaik untuk inspeksi manual

## Fase 5 — Advanced discriminator (opsional, 3–5 hari)
- [ ] Implementasi `TextureDiscriminator`
- [ ] Integrasi objective `L_adv_tex`
- [ ] Jalankan Run G + bandingkan terhadap best F

## Fase 6 — Hardening & release v2 (2–4 hari)
- [ ] Refactor config (`config.v2.yaml`)
- [ ] Update dokumentasi eksperimen + README inference
- [ ] Finalisasi web app integration notes
- [ ] Tag release internal `hcas-gan-v2`

---

## 9) Struktur Konfigurasi Baru (Usulan)

Contoh blok baru di `config.v2.yaml`:

- `style_prior.enabled`
- `style_prior.dataset_dir`
- `style_prior.lambda_style`
- `palette.enabled`
- `palette.lambda_palette`
- `frequency.enabled`
- `frequency.lambda_freq`
- `curriculum.lambda_sal_start`
- `curriculum.lambda_sal_end`
- `curriculum.warmup_epochs`
- `sampling.mask_area_curriculum`

---

## 10) Deliverables yang Wajib Ada per Fase

1. Kode + config commit terpisah per fase.
2. Catatan eksperimen ringkas (tujuan, perubahan, hasil).
3. Artifact visual standar:
   - perbandingan environment/mask/pattern/composite,
   - flat pattern 1x1.
4. Tabel metrik komparatif lintas run.

---

## 11) Keputusan Go/No-Go

Go ke fase lanjutan jika memenuhi:

- Skor visual military-likeness naik signifikan dari baseline.
- Tidak ada degradasi besar pada anti-saliency.
- Training stabil dan reproducible (seed tetap).

Jika tidak terpenuhi, rollback ke fase sebelumnya dan lakukan hyperparameter sweep terbatas.

---

## 12) Penutup

Roadmap v2 ini sengaja dibuat **iteratif**: mulai dari quick-win low-risk, lalu naik ke komponen advanced. Dengan pendekatan ini, tim bisa melihat peningkatan bertahap yang terukur tanpa “big bang rewrite”.

Prinsip utamanya: **style realism + saliency suppression + experiment discipline**.
