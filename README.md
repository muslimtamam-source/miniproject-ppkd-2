# miniproject-ppkd-2
Chatbot_Sales_Marketing_Smartphone

# Ponsel Studio — Chatbot Sales Ponsel

Ponsel Studio adalah chatbot berbasis **Streamlit** untuk membantu pelanggan mencari ponsel, memeriksa harga dan stok, meminta rekomendasi, serta menghitung simulasi belanja. Aplikasi membaca katalog CSV di folder `knowledge_docs/`. Untuk pertanyaan deskriptif, aplikasi menggunakan pendekatan **Retrieval-Augmented Generation (RAG)** dengan LangChain, Chroma, dan model chat melalui Groq.

> **Status proyek:** katalog yang dicontohkan berisi data simulasi untuk pembelajaran. Harga, tahun, spesifikasi, dan stok pada CSV contoh tidak boleh dianggap sebagai penawaran toko atau informasi pasar terkini. Simulasi belanja belum membuat transaksi dan tidak mengurangi stok.

## 1. Latar belakang dan tujuan

Pelanggan sering bertanya dengan bahasa yang berbeda-beda: “Ada iPhone apa saja?”, “Berapa total stok?”, “Ponsel mana yang sesuai anggaran?”, atau “Berapa total jika membeli beberapa unit?”. Pencarian RAG hanya mengambil beberapa potongan data yang relevan. Karena itu, aplikasi memisahkan pekerjaan menjadi dua jalur:

- **Perhitungan dan fakta katalog:** Python membaca seluruh baris CSV untuk daftar merek, model, harga, stok, peringkat harga, pengelompokan, anggaran, dan subtotal pesanan.
- **Penjelasan dan rekomendasi:** RAG mengambil konteks produk, lalu model chat menyusun jawaban dalam bahasa Indonesia sesuai `system_prompt.md`.

Tujuannya adalah memberi jawaban yang mudah dipahami pelanggan sambil menjaga angka katalog tetap sesuai data. Manfaatnya bagi demonstrasi adalah alur RAG terlihat jelas, sedangkan pertanyaan hitung dapat diperiksa ulang dari CSV.

## 2. Sumber data dan struktur proyek

Dataset contoh: `Sample_Dataset_Stok_Harga_Ponsel_100_Baris(1).csv`. Pada versi yang diperiksa, terdapat **100 baris model, 20 merek, dan 2.826 unit stok**. Angka tersebut berasal dari penjumlahan kolom `Jumlah Stok`; angka di dashboard akan berubah jika CSV diganti.

| Kolom CSV | Kegunaan |
| --- | --- |
| `ID Barang` | Pengenal produk untuk pencarian tepat |
| `Merk/Type` | Nama merek dan model yang ditampilkan |
| `Spesifikasi` | Bahan jawaban deskriptif dan pengelompokan RAM/penyimpanan |
| `Tahun Pembuatan` | Pengelompokan stok menurut tahun |
| `Harga` | Harga satuan dalam rupiah, tanpa simbol `Rp` di CSV |
| `Jumlah Stok` | Banyaknya unit yang tersedia pada baris produk |

Susun berkas seperti ini di folder proyek:

```text
proyek-chatbot-ponsel/
├── app.py
├── rag_chatbot.py
├── system_prompt.md
├── .env                         # dibuat sendiri; jangan unggah ke GitHub
└── knowledge_docs/
    └── Sample_Dataset_Stok_Harga_Ponsel_100_Baris.csv
```

`app.py` membaca semua `*.csv` dalam `knowledge_docs/` dengan encoding `utf-8-sig`. File PDF dalam folder yang sama juga dapat dimuat melalui fungsi PDF di `rag_chatbot.py`. Untuk demonstrasi katalog ini, cukup gunakan CSV. Jangan menaruh salinan CSV yang sama dua kali dalam folder tersebut karena baris dan stok akan terhitung dua kali. Nama file CSV bebas; header kolom harus sesuai tabel di atas. Data contoh memakai pemisah koma, satu produk per baris.

## 3. Instalasi di VS Code (Windows PowerShell)

1. Buka folder proyek di VS Code, lalu buka **Terminal > New Terminal**. Pastikan terminal berada pada folder yang berisi `app.py`.
2. Aktifkan environment Python yang akan digunakan. Bila sudah memiliki Conda environment `PPKD26`:

   ```powershell
   conda activate PPKD26
   ```

   Alternatif bila tidak memakai Conda, buat virtual environment baru:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Instal dependensi yang dipakai oleh kedua script:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install streamlit python-dotenv langchain-core langchain-groq langchain-pymupdf4llm langchain-text-splitters langchain-chroma
   ```

   `langchain-chroma` memasang dependensi Chroma yang dibutuhkannya. Jika proyek mempunyai `requirements.txt` yang telah diuji dan dikunci versinya, gunakan `python -m pip install -r requirements.txt` sebagai pengganti daftar paket di atas. Periksa environment VS Code dengan `python -c "import streamlit, langchain_chroma, langchain_groq; print('Import OK')"`.

4. Buat file `.env` di folder yang sama dengan `app.py`:

   ```dotenv
   GROQ_API_KEY=isi_api_key_anda
   ```

   Dapatkan API key dari akun Groq Anda. **Jangan tulis key asli di README, CSV, kode, atau commit Git.** Masukkan `.env`, `.venv/`, dan `__pycache__/` ke `.gitignore`. Pada deployment, pastikan `GROQ_API_KEY` tersedia sebagai environment variable karena `app.py` memeriksanya melalui `os.getenv()`.

5. Pastikan CSV ada di `knowledge_docs/`, lalu jalankan:

   ```powershell
   python -m streamlit run app.py
   ```

   Buka alamat lokal yang ditampilkan terminal, biasanya `http://localhost:8501`. Pada pemuatan awal, aplikasi membangun indeks vektor. Sesudah CSV diubah, aplikasi membaca data dan memperbarui cache indeks berdasarkan ukuran serta waktu modifikasi file.

**Penting:** jalankan `app.py` untuk chatbot sales CSV. Fungsi `main()` pada `rag_chatbot.py` yang berdiri sendiri masih merupakan contoh lama berbasis **PDF** dan belum diselaraskan untuk menjalankan katalog CSV langsung. `app.py` mengimpor fungsi RAG yang diperlukan dari file tersebut. Komentar lama tentang RUU dan nama collection `ruu_ketenagakerjaan` di `rag_chatbot.py` adalah sisa penamaan contoh awal; nama itu tidak menentukan isi jawaban pelanggan ketika aplikasi menggunakan CSV dan `system_prompt.md` terbaru.

## 4. Model dan arsitektur

| Komponen | Implementasi dalam kode |
| --- | --- |
| Antarmuka | Streamlit pada `app.py` |
| Model chat | `openai/gpt-oss-120b` melalui `ChatGroq`, `temperature=0`, `reasoning_effort="low"` |
| Prompt | `system_prompt.md`, peran sales yang profesional dan batasan agar tidak mengarang data |
| Pembaca CSV | `csv.DictReader` pada `baca_produk()`; setiap baris menjadi `Document` |
| Pembaca PDF opsional | `PyMuPDF4LLMLoader` pada `rag_chatbot.py` |
| Chunking | `RecursiveCharacterTextSplitter`, `chunk_size=500`, `chunk_overlap=50` |
| Embedding dan vector store | `Chroma` melalui `langchain-chroma`; kode **tidak menetapkan embedding model secara eksplisit**, sehingga konfigurasi/default library yang terpasang perlu diperiksa pada environment saat digunakan |
| Retrieval | `vectorstore.as_retriever(search_kwargs={"k": TOP_K})` dengan `TOP_K=3` |
| Generation | `ChatPromptTemplate` menggabungkan konteks dan pertanyaan, lalu `ChatGroq` menghasilkan respons dan `StrOutputParser` mengambil teks |

Alur RAG untuk pertanyaan deskriptif:

```text
CSV/PDF → Document → chunking → embedding → Chroma
Pertanyaan → retrieval 3 potongan → konteks + prompt → model Groq → jawaban
```

Alur untuk angka katalog:

```text
Pertanyaan → fungsi pencocokan dan perhitungan Python → seluruh CSV → jawaban angka
```

Pemisahan ini penting. Menjumlahkan tiga hasil retrieval tidak menghasilkan total stok dari 100 baris. Kode `jawaban_data()` dan `jawaban_perhitungan()` menangani pertanyaan katalog lebih dahulu. Jika tidak ada jawaban langsung yang cocok, `app.py` memanggil RAG.

## 5. Cara kerja dan contoh input-output

1. `baca_produk()` membaca CSV, membersihkan nilai, lalu memvalidasi harga dan stok sebagai bilangan tidak negatif.
2. `muat_dokumen_produk()` mengubah setiap baris menjadi `Document` dengan metadata sumber dan nomor baris. PDF opsional digabungkan.
3. `bangun_vectorstore()` membagi dokumen menjadi chunk, membuat indeks Chroma, dan menyediakan retriever.
4. Saat pelanggan mengirim pertanyaan, aplikasi mencoba hitungan langsung terlebih dahulu. Pencarian merek mengabaikan kapitalisasi dan dapat mengenali salah ketik ringan, misalnya `IPhone` atau `iphon`.
5. Pertanyaan rekomendasi yang tidak ditangani aturan hitung diteruskan ke RAG. Prompt membatasi jawaban pada informasi yang ada dalam konteks.

Contoh dari CSV yang diperiksa:

| Pertanyaan pelanggan | Jawaban yang diharapkan |
| --- | --- |
| `Merek apa saja yang tersedia?` | Daftar 20 merek dari seluruh CSV |
| `Berapa total stok semua ponsel?` | 2.826 unit dari 100 model |
| `Daftar 5 harga tertinggi` | Urutan harga menurun; iQOO 12 berada di atas dengan Rp11.424.000 |
| `Berapa stok iPhone?` | 113 unit dari 5 model iPhone |
| `Beli 2 iPhone 11 dan 3 Samsung Galaxy A15` | 2 × Rp1.674.000 + 3 × Rp2.499.000 = **Rp10.845.000** |
| `Stok iPhone 99?` | Model dengan nomor tersebut tidak ditemukan pada katalog |

Pertanyaan lain yang didukung meliputi stok per merek/tahun/RAM/penyimpanan, harga tertinggi dan terendah, rata-rata harga per model, nilai persediaan berdasarkan harga jual, stok kosong, dan pilihan dalam batas anggaran. Diskon hanya dihitung sebagai **asumsi simulasi** bila pelanggan menyebutkan persentasenya. Ongkir, pajak, promo resmi, dan pemrosesan transaksi belum tersedia. Bila jumlah pesanan melebihi stok, aplikasi tetap menampilkan hitungan tetapi memberi peringatan stok tidak cukup.

## 6. Pengujian dan evaluasi

Gunakan setidaknya enam pertanyaan pada tabel di atas saat demo. Bandingkan angka dengan CSV menggunakan penjumlahan atau filter sederhana. Periksa juga dua kasus tambahan:

- Salah ketik: `Berapa stok iphon 11?` harus mengarah ke model iPhone 11 yang tepat.
- Stok tidak cukup: `Beli 100 iPhone 11` harus menampilkan peringatan karena CSV contoh mencatat 47 unit iPhone 11.

Pada pengujian fungsi Python menggunakan CSV contoh, daftar 20 merek, 100 model, total 2.826 unit, subtotal multi model, dan contoh kesalahan input di atas telah diperiksa. **Jawaban RAG melalui API Groq belum dievaluasi otomatis dalam pengujian tersebut**. Saat demo langsung, cek apakah rekomendasi hanya menyebut atribut yang benar-benar tersedia pada konteks dan tidak mengarang detail kamera, garansi, atau promo.

Rencana demo selama **5–7 menit**:

1. Menit 0–1: tunjukkan dashboard dan sumber CSV.
2. Menit 1–2,5: tampilkan semua merek dan harga tertinggi/terendah.
3. Menit 2,5–4: cek total stok dan pengelompokan per merek.
4. Menit 4–5,5: hitung simulasi belanja beberapa model.
5. Menit 5,5–7: uji salah ketik, model tidak ditemukan, lalu satu pertanyaan rekomendasi RAG.

## 7. Estimasi biaya

Kode menetapkan model chat `openai/gpt-oss-120b` melalui Groq. Berdasarkan [halaman model resmi Groq](https://console.groq.com/docs/model/openai/gpt-oss-120b) dan [halaman harga Groq](https://groq.com/pricing) yang diperiksa **27 September 2026**, tarif on-demand yang ditampilkan adalah **US$0,15 per 1 juta token input** dan **US$0,60 per 1 juta token output**. Tarif dapat berubah; periksa kembali sebelum digunakan atau dipresentasikan.

Contoh estimasi untuk **satu pertanyaan yang benar-benar memanggil Groq** dengan asumsi 2.000 token input dan 300 token output:

```text
Input  = 2.000 / 1.000.000 × US$0,15 = US$0,00030
Output =   300 / 1.000.000 × US$0,60 = US$0,00018
Total estimasi per pertanyaan            = US$0,00048
100 pertanyaan dengan ukuran yang sama   = US$0,048
```

Angka ini **contoh perhitungan**, bukan tagihan aktual. Jumlah token nyata bergantung pada panjang prompt, potongan dokumen, pertanyaan, dan jawaban. Pertanyaan yang selesai dihitung langsung oleh Python tidak mengirim pertanyaan itu ke model Groq, sehingga tidak menambah token generation Groq. Estimasi di atas belum memasukkan biaya hosting, listrik/perangkat, penyimpanan, dan kemungkinan layanan embedding lain jika konfigurasi diubah. Aplikasi saat ini tidak mencatat pemakaian token secara terpisah.

## 8. Batasan penting

- CSV adalah snapshot. Aplikasi belum tersambung ke inventaris toko secara langsung. Perubahan harga/stok memerlukan perubahan data sumber.
- Pengelompokan merek memakai token pertama dari `Merk/Type`, dengan `Apple iPhone` ditampilkan sebagai `iPhone`. Jika format nama produk berubah, aturan `nama_merek()` perlu disesuaikan.
- Salah ketik ringan dapat ditangani, tetapi nama model yang ambigu perlu diklarifikasi; jangan menganggap pencocokan otomatis selalu tepat.
- Riwayat percakapan disimpan selama sesi Streamlit. Simulasi pesanan terakhir dapat digunakan untuk pertanyaan lanjutan sederhana seperti “totalnya berapa?”, tetapi aplikasi belum mempunyai keranjang belanja permanen.
- `rag_chatbot.py` belum menetapkan embedding model secara eksplisit dan masih menyimpan komentar serta nama collection dari proyek RUU lama. Untuk laporan teknis, sebutkan keadaan kode yang sebenarnya.
- PDF opsional dapat memberi konteks tambahan, tetapi harga dan stok yang dihitung aplikasi berasal dari CSV.

## 9. Berkas yang dikumpulkan

- `app.py`: antarmuka Streamlit, pembacaan CSV, perhitungan katalog, dan jalur pertanyaan.
- `rag_chatbot.py`: model chat, loader PDF opsional, chunking, vector store, retrieval, dan RAG chain.
- `system_prompt.md`: aturan perilaku sales ponsel dan batasan informasi.
- `knowledge_docs/*.csv`: dataset contoh yang digunakan saat demonstrasi.
- Presentasi: latar belakang, tujuan, manfaat, arsitektur, cara kerja, input-output, dan evaluasi.
- README ini: instalasi, penggunaan, model, sumber data, estimasi biaya, batasan, dan rencana demo.
