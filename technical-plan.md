# SPESIFIKASI TEKNIS: Arsitektur HCAS-GAN

**Hybrid Contextual Anti-Saliency Generative Adversarial Network**

- **Versi**: 2.1.0 (Extended Implementation Ready)
- **Status**: Rancangan Arsitektur Tingkat Lanjut & Panduan Rekayasa Perangkat Lunak

---

## 1. Pendahuluan dan Latar Belakang Arsitektur

Arsitektur HCAS-GAN merupakan perpaduan kompleks antara pemodelan generatif dan psikofisika komputasional. Berbeda dengan model GAN tradisional yang hanya bertugas meniru distribusi data (seperti membuat wajah manusia yang tidak ada), HCAS-GAN memiliki objektif fungsional ganda:

### Objektif Dual-Layer

1. **Penyatuan Struktural (Structural Blending)**
   - Menghasilkan tekstur yang secara piksel dan frekuensi spasial identik dengan lingkungan sekitarnya (hutan, gurun, urban)

2. **Penghindaran Atensi (Attention Evasion)**
   - Secara aktif memanipulasi fitur visual tingkat tinggi untuk menipu Human Visual System (HVS) agar tidak mengunci fokus (fiksasi) pada target yang dikamuflasekan

Untuk mencapai hal ini, sistem mengintegrasikan DeepGaze III (sebagai proksi biologis mata manusia) ke dalam siklus optimasi (training loop) dari arsitektur PatchGAN.## 2. Tumpukan Teknologi (Tech Stack) & Dependensi Lanjutan

Pemilihan framework dalam proyek ini tidak hanya didasarkan pada preferensi, melainkan pada kebutuhan matematis arsitektur. Penggunaan fungsi kerugian (loss function) kustom yang melibatkan inferensi dari model pihak ketiga (DeepGaze) membutuhkan fleksibilitas tingkat tinggi.

### Framework dan Rastaianalisasi

#### Core Machine Learning Framework: PyTorch

- **Library**: `torch` (PyTorch 2.x) dan `torchvision`
- **Rasionalisasi**: PyTorch menggunakan Komputasi Graf Dinamis (Dynamic Computational Graph) / Eager Execution. Ini sangat krusial karena operasi masking spasial dan perhitungan penalti saliensi pada HCAS-GAN berubah-ubah dimensi dan posisinya pada setiap iterasi batch. PyTorch memungkinkan pelacakan gradien (autograd) menembus operasi tensor kustom ini tanpa overhead kompilasi statis.

#### Saliency Modeling Library

- **Library**: `pysaliency` (Bethge Lab)
- **Rasionalisasi**: Library ini bukan sekadar pembungkus model, melainkan menangani normalisasi input spesifik yang dibutuhkan oleh DeepGaze III (seperti penyesuaian center bias dan spatial frequency). DeepGaze III mengekstraksi fitur menggunakan backbone ResNet/DenseNet yang sangat sensitif terhadap prapemrosesan gambar.

#### Computer Vision & Matrix Operations

- **Library**: `opencv-python` (cv2), `numpy`, `scikit-image`
- **Rasionalisasi**: Digunakan untuk manipulasi masking poligon presisi tinggi, ekstraksi kontur siluet target, dan kalkulasi metrik perseptual pasca-pelatihan.

#### MLOps & Experiment Tracking

- **Library**: `wandb` (Weights & Biases)
- **Rasionalisasi**: Melatih GAN sangat rentan terhadap Mode Collapse (kegagalan Generator menghasilkan variasi). wandb digunakan untuk melacak divergensi kerugian Generator vs. Discriminator, memantau distribusi gradien secara real-time, dan secara otomatis mengunggah sampel gambar komposit setiap 50 epoch untuk inspeksi visual ("arms race log").## 3. Struktur Repositori Skala Produksi (Project Directory)

Struktur direktori ini dirancang untuk memfasilitasi kerja tim lintas disiplin (Data Engineer, ML Researcher, dan DevOps) dengan memisahkan logika model dari alur pemrosesan data.

```
hcas-gan-project/
│
├── dataset/                    # Direktori data (Tidak masuk Git)
│   ├── raw_environments/       # Gambar asli (resolusi tinggi 4K)
│   ├── masks/                  # Binary masks (siluet berbagai pose/kendaraan)
│   └── processed/              # Gambar yang sudah di-crop & di-resize ke 256x256
│
├── src/                        # Kode sumber komputasional utama
│   ├── __init__.py
│   ├── models/                 # Definisi Arsitektur Neural Network
│   │   ├── __init__.py
│   │   ├── generator.py        # Arsitektur U-Net 256 (Fokus pada Skip Connections)
│   │   ├── discriminator.py    # Arsitektur PatchGAN 70x70
│   │   └── deepgaze_wrapper.py # Integrasi pysaliency dengan PyTorch autograd
│   │
│   ├── data/                   # Logika asupan data
│   │   ├── augmentations.py    # Rotasi mask, color jitter, spatial scaling
│   │   └── dataset_loader.py   # PyTorch DataLoader (Multiprocessing)
│   │
│   ├── training/               # Logika alur pelatihan
│   │   ├── loss.py             # Formulasi HCAS Loss (Adversarial + Saliency)
│   │   ├── scheduler.py        # Learning Rate Decay (Cosine Annealing)
│   │   └── trainer.py          # Training loop, Forward/Backward pass
│   │
│   └── utils/                  # Skrip evaluasi dan utilitas
│       ├── metrics_ssim.py     # Structural Similarity Index
│       └── metrics_lpips.py    # Learned Perceptual Image Patch Similarity
│
├── requirements.txt            # Dependensi ketat (pip freeze)
├── config.yaml                 # Master node hyperparameter
└── train.py                    # Entry point eksekusi sistem
```

### Implikasi Struktur

Pemisahan `models/` dan `training/` memungkinkan periset mengganti arsitektur Generator (misalnya dari U-Net beralih ke Vision Transformers) tanpa merusak skrip pelatihan.## 4. Implementasi Kode Inti dan Matematika Komputasi

Bagian paling rawan kegagalan matematis dalam arsitektur HCAS-GAN adalah saat merumuskan fungsi penalti atensi. Jika diformulasikan salah, nilai gradien dapat meledak (exploding gradients) atau sistem memanipulasi kelemahan komputasi.

### 4.1 HCAS Loss Function (loss.py)

Kita menggunakan `BCEWithLogitsLoss` alih-alih `BCELoss` murni karena ia menggabungkan lapisan Sigmoid dan Binary Cross Entropy dalam satu kelas, memberikan stabilitas numerik yang jauh lebih baik (mencegah Log(0) saat Discriminator terlalu yakin).

```python
import torch
import torch.nn as nn

class HCASLoss(nn.Module):
    def __init__(self, lambda_sal=10.0):
        super(HCASLoss, self).__init__()
        # lambda_sal mengontrol seberapa agresif sistem menekan atensi visual
        self.lambda_sal = lambda_sal
        self.adversarial_loss = nn.BCEWithLogitsLoss() 

    def compute_saliency_penalty(self, saliency_map, mask):
        """
        Menghitung penalti perseptual. Saliency map direduksi menjadi
        sebuah skalar yang merepresentasikan 'tingkat keterlihatan' target.
        """
        # 1. Isolasi Area: Hadamard Product memastikan kita hanya menghitung
        # probabilitas atensi yang jatuh TEPAT pada siluet kamuflase.
        attention_on_target = saliency_map * mask
        
        # 2. Normalisasi Luas: Siluet bisa berukuran kecil atau besar. 
        # Tambahan 1e-8 adalah konstanta Epsilon untuk mencegah 'Division by Zero'
        # jika secara tidak sengaja ada mask yang kosong sepenuhnya.
        target_area_sum = torch.sum(mask, dim=(1, 2, 3)) + 1e-8
        total_attention = torch.sum(attention_on_target, dim=(1, 2, 3))
        
        # 3. Rata-rata intensitas atensi spesifik per piksel target
        mean_saliency_penalty = torch.mean(total_attention / target_area_sum)
        return mean_saliency_penalty

    def generator_loss(self, discriminator_pred, saliency_map, mask):
        """
        Menghitung total objektif Generator: 
        L_G = L_adv + (lambda * L_sal)
        """
        # Objektif 1: Evaluasi Struktural (Menipu PatchGAN)
        # Kita melatih Generator untuk mendorong output PatchGAN ke nilai 1 (Real)
        valid_labels = torch.ones_like(discriminator_pred)
        loss_adv = self.adversarial_loss(discriminator_pred, valid_labels)
        
        # Objektif 2: Evaluasi Kognitif (Menipu DeepGaze)
        loss_sal = self.compute_saliency_penalty(saliency_map, mask)
        
        # Penjumlahan dinamis. Jika lambda_sal terlalu tinggi, pola akan
        # kehilangan tekstur alamiah demi menghindari atensi.
        total_g_loss = loss_adv + (self.lambda_sal * loss_sal)
        return total_g_loss, loss_adv, loss_sal
```

### 4.2 Komposisi Gambar Dinamis dan Forward Pass (trainer.py)

Proses komposit gambar di bawah ini krusial. Sistem menyimulasikan "pemasangan" seragam pada target di lapangan secara virtual.

```python
# Blok kode di dalam iterasi epoch DataLoader
# z (Noise), B (Background Asli), M (Mask Siluet Target)

# 1. SINTESIS TEKSTUR: U-Net mengekstraksi fitur latar belakang dan
# memadukannya dengan noise untuk menghasilkan desain yang bervariasi.
fake_pattern = generator(B, z)

# 2. KOMPOSISI FISIK (Alpha Blending dengan Mask):
# Bagian luar mask mempertahankan latar belakang asli.
# Bagian dalam mask diisi oleh hasil generator.
composite_image = (B * (1.0 - M)) + (fake_pattern * M)

# 3. EVALUASI KOGNITIF (Human Visual System):
# torch.no_grad() SANGAT KRUSIAL! Kita tidak ingin mengubah bobot DeepGaze.
# Jika ini tidak dilakukan, GPU memori akan habis seketika karena menyimpan
# graf komputasi dari model DeepGaze yang masif.
with torch.no_grad():
    # Mengembalikan tensor probabilitas fiksasi mata (Heatmap)
    saliency_heatmap = model_deepgaze(composite_image)

# 4. EVALUASI STRUKTURAL (PatchGAN Discriminator):
# Discriminator memeriksa batas antara kamuflase dan latar belakang.
# Apakah terlihat seperti editan (stitching) atau menyatu secara natural?
disc_prediction = discriminator(composite_image)

# 5. BACKPROPAGATION (Pembaruan Bobot Generator):
# Menghitung seberapa buruk performa Generator, menghitung turunan parsial
# (gradien), dan memperbarui bobot U-Net.
loss_G, l_adv, l_sal = hcas_loss.generator_loss(disc_prediction, saliency_heatmap, M)
optimizer_G.zero_grad()
loss_G.backward()
optimizer_G.step()
```
## 5. Master Node Konfigurasi (config.yaml)

Sentralisasi parameter ini mencegah hard-coding di dalam logika pelatih. Nilai-nilai di bawah ini diatur spesifik untuk stabilitas DCGAN.

```yaml
experiment:
  name: "HCAS_GAN_Jungle_Dense_v2"
  seed: 42                # Untuk reproduktibilitas hasil

hardware:
  device: "cuda"          # Wajib menggunakan GPU
  num_workers: 8          # Multiprocessing untuk memuat gambar

hyperparameters:
  learning_rate_G: 0.0002 # Standard rule-of-thumb GAN
  learning_rate_D: 0.0002 # Menggunakan Two-Time-Scale Update Rule (TTUR) jika perlu
  beta1: 0.5              # Sangat krusial. Mencegah osilasi mematikan pada Adam Optimizer
  beta2: 0.999
  batch_size: 16          # Diturunkan jika VRAM < 16GB
  epochs: 1000            # Pelatihan panjang dengan early stopping

hcas_specific:
  lambda_sal: 15.0        # Bobot prioritas anti-saliensi vs realisme tekstur
  image_size: 256         # Kompromi ideal antara detail makro/mikro kamuflase
  latent_dim: 100         # Dimensi variabilitas vektor noise Z

augmentations:
  random_mask_rotation: true
  mask_scale_range: [0.3, 0.8] # Ukuran siluet relatif terhadap background
```

### Penjelasan Parametrik Khusus

**Parameter beta1: 0.5**

Parameter `beta1 = 0.5` pada Adam Optimizer adalah keharusan mutlak dalam melatih Discriminator. Angka momentum yang terlalu tinggi (seperti 0.9 pada klasifikasi citra standar) menyebabkan model berjalan melewati local minima dan membuat kompetisi antara Generator dan Discriminator berosilasi tak terkendali.## 6. Kebutuhan Perangkat Keras dan Performa

Karena HCAS-GAN secara simultan memuat Tiga Model Jaringan Syaraf di dalam memori (U-Net Generator, PatchGAN Discriminator, dan DeepGaze III Saliency Model), spesifikasi perangkat keras menjadi tantangan utama.

### GPU VRAM (Memori Video)

Mutlak membutuhkan minimal **16GB VRAM** (Rekomendasi: NVIDIA RTX 4080, RTX 3090, atau A100 untuk level enterprise). Mengalokasikan DeepGaze III ke memori membutuhkan ruang besar bahkan saat modenya `eval()`.

### Mitigasi untuk VRAM Terbatas

Jika VRAM terbatas, mitigasi yang bisa dilakukan adalah:
- Menerapkan Gradient Accumulation
- Menurunkan parameter `batch_size` ke angka 4 atau 8

---

## 7. Metrik Evaluasi Akhir (Post-Training)

Selain loss visual, desain yang disintesis harus diuji melalui metrik saintifik.

### SSIM (Structural Similarity Index)

- **Kegunaan**: Digunakan pasca-pelatihan pada patch hasil Generator yang diisolasi dengan patch alam asli
- **Target nilai**: > 0.75
- **Manfaat**: Menjamin kamuflase menyerupai struktur organik

### LPIPS (Learned Perceptual Image Patch Similarity)

- **Kegunaan**: Mengukur seberapa jauh perbedaan tekstur mikro dengan latar belakang dalam "pandangan" lapisan awal model VGG
- **Manfaat**: Mencegah Generator mencetak noise acak yang menipu metrik tradisional

---

## 8. Glossarium (Awam + Expert)

Bagian ini disusun agar dokumen dapat dipahami oleh dua audiens sekaligus:
- **Awam**: fokus pada analogi dan fungsi praktis.
- **Expert**: fokus pada definisi teknis, implikasi matematis, dan implementasi.

### 8.1 Konsep Dasar AI, ML, dan Deep Learning

| No | Istilah | Penjelasan Awam | Penjelasan Expert |
|---:|---|---|---|
| 1 | Artificial Intelligence (AI) | Teknologi agar komputer bisa melakukan tugas yang biasanya butuh kecerdasan manusia. | Bidang komputasi yang mencakup symbolic reasoning, search, optimization, dan machine learning untuk decision-making otomatis. |
| 2 | Machine Learning (ML) | Cara membuat komputer belajar dari contoh data, bukan dari aturan manual satu per satu. | Paradigma statistik untuk mempelajari fungsi $f_\theta(x)$ dari data agar meminimalkan objective pada distribusi target. |
| 3 | Deep Learning | Sub-bidang ML yang memakai jaringan saraf bertingkat untuk mengenali pola kompleks. | ML berbasis neural network multi-layer dengan representasi hierarkis non-linear yang dilatih via backpropagation. |
| 4 | Computer Vision | Cabang AI agar komputer bisa “melihat” dan memahami gambar/video. | Pemodelan informasi visual (2D/3D/temporal/semantik) untuk detection, segmentation, recognition, dan synthesis. |
| 5 | Neural Network | Model komputasi yang meniru cara kerja neuron sederhana untuk mengolah informasi. | Komposisi fungsi linear + non-linear berparameter $\theta$ yang dioptimasi terhadap loss. |
| 6 | CNN (Convolutional Neural Network) | Jenis jaringan saraf yang sangat efektif untuk gambar. | Arsitektur berbasis convolution kernel, local receptive field, weight sharing, dan translational equivariance. |

### 8.2 Istilah Inti GAN dan Arsitektur HCAS-GAN

| No | Istilah | Penjelasan Awam | Penjelasan Expert |
|---:|---|---|---|
| 7 | GAN (Generative Adversarial Network) | Dua model AI yang “berkompetisi”: satu membuat gambar, satu menilai keaslian gambar. | Framework minimax antara Generator $G$ dan Discriminator $D$ untuk mendekati distribusi data nyata $p_{data}$. |
| 8 | Generator (G) | “Seniman” yang membuat pola kamuflase baru. | Model parametrik pemetaan input kondisional/noise ke sample sintetis untuk memaksimalkan fooling terhadap $D$. |
| 9 | Discriminator (D) | “Juri” yang menilai gambar asli atau hasil generator. | Binary classifier berbasis logits untuk memisahkan real vs fake. |
| 10 | PatchGAN | Penilai gambar yang mengecek keaslian per bagian kecil, bukan hanya keseluruhan gambar. | Discriminator lokal yang memodelkan realism tingkat patch (mis. 70×70), efektif untuk detail tekstur. |
| 11 | U-Net | Arsitektur yang menjaga detail gambar lewat jalur turun-naik fitur. | Encoder-decoder simetris dengan skip connections untuk pelestarian detail spasial multi-skala. |
| 12 | Skip Connection | “Jalur pintas” agar informasi detail tidak hilang saat jaringan makin dalam. | Koneksi identitas/concatenation lintas layer yang memperbaiki gradient flow dan feature reuse. |
| 13 | Latent Vector / Noise ($z$) | Sumber variasi acak agar hasil kamuflase tidak monoton. | Variabel laten dari prior distribution (Gaussian/Uniform) untuk kontrol mode sintetis. |
| 14 | Conditional Generation | Generasi gambar yang mengikuti kondisi tertentu (mis. latar belakang tertentu). | Estimasi distribusi bersyarat $p(x|c)$ dengan kondisi $c$ (image/mask/label/embedding). |
| 15 | Mode Collapse | Generator hanya menghasilkan tipe output yang itu-itu saja. | Degenerasi training GAN saat support distribusi output menyempit dan gagal menangkap keragaman $p_{data}$. |

### 8.3 Istilah Spesifik HCAS-GAN (Kamuflase + Atensi)

| No | Istilah | Penjelasan Awam | Penjelasan Expert |
|---:|---|---|---|
| 16 | Structural Blending | Pola kamuflase menyatu dengan tekstur lingkungan sekitar. | Kesesuaian statistik spasial/frekuensi antara region sintetis dan background lokal. |
| 17 | Attention Evasion | Strategi agar mata manusia tidak cepat tertarik ke objek target. | Optimasi fitur visual untuk menurunkan probabilitas fiksasi pada ROI target berbasis saliency proxy. |
| 18 | Human Visual System (HVS) | Sistem biologis manusia untuk melihat dan fokus pada objek. | Proses penglihatan retina–korteks yang melibatkan bottom-up saliency dan top-down attention. |
| 19 | Fixation (Fiksasi Mata) | Titik yang menjadi fokus pandangan mata. | Lokasi gaze dwell yang dipakai sebagai target/ground truth pada saliency modeling. |
| 20 | Saliency Map | Peta panas area gambar yang paling mungkin dilihat manusia terlebih dahulu. | Distribusi probabilitas perhatian visual per piksel/patch. |
| 21 | DeepGaze III | Model AI yang meniru kecenderungan fokus mata manusia pada gambar. | Model prediksi saliency berbasis deep features + center bias, dilatih dari data fixation manusia. |
| 22 | Saliency Penalty ($L_{sal}$) | Hukuman jika area target terlalu “mencolok” di saliency map. | Regularizer yang mengagregasi saliency pada ROI bermask, biasanya dinormalisasi luas mask. |
| 23 | $\lambda_{sal}$ (Lambda Saliency) | Kenop untuk mengatur seberapa kuat model menekan keterlihatan target. | Hyperparameter trade-off pada objective generator: $L_G = L_{adv} + \lambda_{sal}L_{sal}$. |
| 24 | Mask (Binary Mask / Siluet) | Peta hitam-putih yang menunjukkan area target yang mau dikamuflasekan. | Tensor ROI (0/1 atau soft mask) untuk compositing, weighting loss, dan evaluasi lokal. |
| 25 | Hadamard Product | Perkalian antar piksel pada posisi yang sama. | Element-wise multiplication pada tensor untuk isolasi kontribusi nilai di area mask. |
| 26 | Alpha Blending / Image Compositing | Menggabungkan dua gambar memakai peta mask sebagai pengatur campuran. | Operasi campuran linear: $I_c = B(1-M) + F\cdot M$. |

### 8.4 Loss, Optimisasi, dan Stabilitas Numerik

| No | Istilah | Penjelasan Awam | Penjelasan Expert |
|---:|---|---|---|
| 27 | Loss Function | Angka kesalahan yang memberi tahu seberapa buruk performa model. | Objective scalar terdiferensiasi yang dioptimasi terhadap parameter model. |
| 28 | Adversarial Loss ($L_{adv}$) | Hukuman saat generator gagal menipu discriminator. | Komponen objective berbasis prediksi real/fake logits dari $D$ untuk melatih $G$ dan $D$. |
| 29 | BCE (Binary Cross Entropy) | Rumus untuk menilai benar-salah prediksi dua kelas (real/fake). | Negative log-likelihood Bernoulli untuk klasifikasi biner. |
| 30 | BCEWithLogitsLoss | Versi BCE yang lebih stabil karena sigmoid sudah dipaketkan di dalam rumus. | Kombinasi sigmoid + BCE dengan formulasi numerik stabil (log-sum-exp trick). |
| 31 | Sigmoid | Fungsi yang mengubah nilai bebas menjadi rentang 0 sampai 1. | Aktivasi $\sigma(x)=\frac{1}{1+e^{-x}}$ untuk memetakan logits ke probabilitas. |
| 32 | Gradient (Gradien) | Arah perubahan parameter agar model makin baik. | Turunan parsial loss terhadap parameter untuk update optimizer. |
| 33 | Backpropagation | Proses menghitung “salahnya model” dari output ke layer awal. | Reverse-mode automatic differentiation untuk gradien efisien pada komputasi graf. |
| 34 | Autograd | Fitur otomatis yang menghitung turunan tanpa rumus manual. | Engine diferensiasi otomatis PyTorch berbasis dynamic graph. |
| 35 | Exploding Gradient | Gradien terlalu besar sehingga training jadi tidak stabil. | Magnitudo gradien tumbuh tak terkendali dan memicu divergensi update parameter. |
| 36 | Epsilon ($1e-8$) | Angka kecil pengaman agar tidak terjadi pembagian dengan nol. | Konstanta stabilisasi numerik dalam operasi rasio/normalisasi. |
| 37 | TTUR (Two-Time-Scale Update Rule) | Strategi memberi kecepatan belajar berbeda untuk Generator dan Discriminator. | Penggunaan learning rate berbeda pada dua pemain GAN untuk stabilitas konvergensi. |
| 38 | Cosine Annealing | Jadwal menurunkan learning rate secara bertahap berbentuk kurva halus. | Scheduler berbasis fungsi cosinus untuk eksplorasi lalu fine-tuning dynamics training. |
| 39 | Adam Optimizer ($\beta_1$, $\beta_2$) | Algoritma update bobot yang adaptif dan umum dipakai di deep learning. | Adaptive moments optimizer; pada GAN, `beta1=0.5` sering dipakai untuk meredam osilasi. |

### 8.5 Data Pipeline, Augmentasi, dan Reproducibility

| No | Istilah | Penjelasan Awam | Penjelasan Expert |
|---:|---|---|---|
| 40 | DataLoader | Komponen untuk memuat data per batch secara otomatis. | Abstraksi iterasi dataset dengan batching, shuffling, multiprocessing, pin memory, dan prefetching. |
| 41 | Batch Size | Jumlah gambar yang diproses sekali jalan oleh model. | Ukuran sampel per step optimisasi; mempengaruhi stabilitas gradien, generalisasi, dan kebutuhan VRAM. |
| 42 | Epoch | Satu putaran penuh model melihat seluruh data latih. | Satuan iterasi global saat seluruh indeks dataset telah di-visit minimal sekali. |
| 43 | Data Augmentation | Memvariasikan data latih agar model tidak mudah overfit. | Transformasi label-preserving untuk memperluas distribusi data efektif. |
| 44 | Random Seed | Angka awal agar hasil eksperimen bisa diulang dengan konsisten. | Inisialisasi deterministik pseudo-random generator lintas library/runtime. |
| 45 | Deterministic Pipeline | Alur data yang hasilnya tetap sama jika dijalankan ulang dengan kondisi sama. | Kontrol sumber non-determinism (RNG, backend kernel, urutan data) untuk reproducibility. |
| 46 | LabelMe Polygon JSON | Format anotasi yang menyimpan titik-titik poligon area objek. | Struktur JSON `shapes[]` berisi `label`, `points`, `shape_type` untuk konversi raster mask. |

### 8.6 Evaluasi, Metrik, dan Monitoring Eksperimen

| No | Istilah | Penjelasan Awam | Penjelasan Expert |
|---:|---|---|---|
| 47 | SSIM | Nilai kemiripan struktur visual antara dua gambar. | Metrik berbasis luminance, contrast, dan structure untuk menilai kesamaan perseptual lokal. |
| 48 | LPIPS | Skor beda gambar berdasarkan persepsi jaringan saraf, bukan sekadar beda piksel mentah. | Jarak fitur deep network (mis. VGG/AlexNet) yang berkorelasi lebih baik dengan persepsi manusia. |
| 49 | Ablation Study | Uji coba mematikan satu komponen untuk melihat seberapa penting komponen itu. | Eksperimen terkontrol untuk mengisolasi kontribusi kausal tiap modul/hyperparameter. |
| 50 | Early Stopping | Menghentikan training saat performa tidak membaik agar hemat waktu dan mencegah overfit. | Kriteria terminasi berbasis plateau/degradasi metrik validasi dengan patience tertentu. |
| 51 | Checkpoint | Simpanan kondisi model di titik tertentu agar bisa lanjut training tanpa mengulang dari awal. | Snapshot state dict model/optimizer/scheduler + metadata step/epoch untuk resume/rollback. |
| 52 | Weights & Biases (W&B) | Platform untuk mencatat grafik training dan contoh hasil model. | Experiment tracking untuk logging metrics/artifacts, config versioning, dan analisis kolaboratif. |

### 8.7 Perangkat Keras dan Efisiensi Komputasi

| No | Istilah | Penjelasan Awam | Penjelasan Expert |
|---:|---|---|---|
| 53 | GPU | Prosesor khusus yang jauh lebih cepat untuk komputasi AI dibanding CPU biasa. | Hardware paralel untuk operasi tensor, terutama dense linear algebra dan convolution. |
| 54 | VRAM | Memori khusus di GPU untuk menampung data dan model saat training. | Memory device accelerator; bottleneck utama untuk batch size, resolusi, dan kedalaman model. |
| 55 | Gradient Accumulation | Trik menumpuk gradien beberapa langkah kecil agar efeknya seperti batch besar. | Simulasi effective batch size besar dengan menunda optimizer step selama $k$ mini-batch. |
| 56 | Mixed Precision (FP16/BF16) | Cara mempercepat training dan menghemat memori dengan presisi angka lebih ringan. | Training precision campuran dengan autocast + gradient scaling untuk throughput tinggi dan stabilitas numerik. |
| 57 | Out-of-Memory (OOM) | Error saat memori GPU/CPU tidak cukup. | Kegagalan alokasi tensor runtime akibat footprint model + activations melebihi kapasitas memori. |

---

### Catatan Penggunaan Glossarium

- Untuk pembaca **awam**, baca kolom “Penjelasan Awam” terlebih dahulu untuk memahami alur konseptual.
- Untuk pembaca **expert**, kolom “Penjelasan Expert” dapat dijadikan acuan desain eksperimen dan debugging.
- Istilah pada glossarium ini dipilih agar konsisten dengan implementasi HCAS-GAN dalam dokumen ini.