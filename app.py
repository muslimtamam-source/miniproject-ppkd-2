import csv
import difflib
import os
import re
import unicodedata
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document

from rag_chatbot import (
    SYSTEM_PROMPT_PATH,
    TOP_K,
    buat_model,
    muat_dokumen,
    bangun_vectorstore,
    muat_system_prompt,
    buat_rag_chain,
)

st.set_page_config(page_title="Ponsel Studio | Asisten Belanja", page_icon="📱", layout="wide")

# DIUBAH: Data CSV dan PDF tetap dicari di folder knowledge_docs.
KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge_docs"


def angka_bulat(nilai):
    """Baca stok dari angka biasa atau pemisah ribuan Indonesia."""
    teks = str(nilai or "").strip().replace(" ", "")
    if not teks:
        return 0
    teks = teks.replace(".", "").replace(",", "")
    return int(teks)


def normalisasi(teks):
    """DIUBAH: Pencarian iPhone, IPhone, dan IPHONE menghasilkan nama yang sama."""
    return unicodedata.normalize("NFKC", str(teks)).casefold().strip()


def baca_produk(folder):
    """DIUBAH: Baca semua CSV sebagai sumber angka stok yang pasti."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder sumber data tidak ditemukan: {folder}")
    produk = []
    for berkas in sorted(folder.glob("*.csv")):
        with berkas.open("r", encoding="utf-8-sig", newline="") as f:
            pembaca = csv.DictReader(f)
            if not pembaca.fieldnames:
                raise ValueError(f"CSV tidak memiliki header kolom: {berkas.name}")
            for nomor_baris, baris in enumerate(pembaca, start=2):
                rapi = {str(k).strip(): str(v or "").strip() for k, v in baris.items() if k is not None}
                if not any(rapi.values()):
                    continue
                nama = rapi.get("Merk/Type", "")
                if not nama:
                    continue
                try:
                    stok = angka_bulat(rapi.get("Jumlah Stok", "0"))
                    harga = angka_bulat(rapi.get("Harga", "0"))
                except ValueError as exc:
                    raise ValueError(f"Angka Harga/Jumlah Stok tidak valid pada {berkas.name} baris {nomor_baris}") from exc
                if stok < 0 or harga < 0:
                    raise ValueError(f"Harga/stok negatif pada {berkas.name} baris {nomor_baris}")
                produk.append({"data": rapi, "nama": nama, "stok": stok, "harga": harga,
                               "source": str(berkas), "row": nomor_baris})
    return produk


def muat_dokumen_produk(folder, produk=None):
    """DIUBAH: Tiap baris CSV menjadi satu dokumen untuk pertanyaan umum RAG."""
    produk = baca_produk(folder) if produk is None else produk
    dokumen = [Document(
        page_content="\n".join(f"{k}: {v}" for k, v in item["data"].items() if v),
        metadata={"source": item["source"], "row": item["row"]},
    ) for item in produk]
    if any(folder.glob("*.pdf")):
        dokumen.extend(muat_dokumen(folder))
    if not dokumen:
        raise FileNotFoundError(f"Tidak ada data CSV/PDF dalam folder: {folder}")
    return dokumen


def rupiah(nilai):
    return "Rp" + f"{nilai:,}".replace(",", ".")


def nama_merek(nama):
    """DIUBAH: iPhone/Apple dikelompokkan dengan label iPhone."""
    if re.search(r"\b(iphone|apple)\b", normalisasi(nama)):
        return "iPhone"
    return nama.split()[0] if nama.split() else nama


def cocok_merek(q, produk):
    """DIUBAH: Temukan merek secara utuh, dengan toleransi salah ketik ringan."""
    semua = {normalisasi(nama_merek(p["nama"])) for p in produk}
    kata = re.findall(r"[\w]+", q)
    ditemukan = {m for m in semua if re.search(r"(?<!\w)" + re.escape(m) + r"(?!\w)", q)}
    if "apple" in kata and "iphone" in semua:
        ditemukan.add("iphone")
    if ditemukan:
        return ditemukan
    # Hanya koreksi kata yang sangat mirip dengan SATU merek agar tidak
    # salah menafsirkan kata umum sebagai nama merek.
    for kata_tanya in kata:
        if len(kata_tanya) < 4:
            continue
        hasil = difflib.get_close_matches(kata_tanya, sorted(semua), n=2, cutoff=0.82)
        if len(hasil) == 1:
            ditemukan.add(hasil[0])
    return ditemukan


def format_produk(p):
    return (f"- **{p['nama']}** · ID {p['data'].get('ID Barang', '—')} · "
            f"{rupiah(p['harga'])} · stok {p['stok']} unit")


def rupiah_dari_teks(q):
    """DIUBAH: Pahami 3 juta, 3 jt, Rp3.000.000, atau 3000000."""
    m = re.search(r"(?:rp\s*)?(\d+(?:[.,]\d+)?)\s*(juta|jt|ribu|rb)\b", q)
    if m:
        angka = float(m.group(1).replace(",", "."))
        return round(angka * (1_000_000 if m.group(2) in {"juta", "jt"} else 1_000))
    m = re.search(r"\brp\s*(\d[\d.,]*)\b", q)
    return angka_bulat(m.group(1)) if m else None


def ringkasan_pesanan(q, produk):
    """DIUBAH: Hitung simulasi pesanan per model dari harga CSV; tanpa checkout."""
    if not re.search(r"\b(beli|pesan|order|checkout|belanja|pembelanjaan|keranjang)\b", q):
        return None
    bagian = re.sub(r"\b(?:diskon|potongan)\s*\d+(?:[.,]\d+)?\s*%", " ", q)
    bagian = re.sub(r"\b(berapa|hitung|total|harga|kalau|jika|saya|ingin|mau|akan|beli|pesan|order|checkout|belanja|pembelanjaan|keranjang|ponsel|hp)\b", " ", bagian)
    segmen = [s.strip() for s in re.split(r"\s+(?:dan|serta|plus)\s+|[+;,]", bagian) if s.strip()]
    pesanan = {}
    masalah = []
    for seg in segmen:
        # Diskon ditangani sesudah subtotal; ia bukan baris produk.
        if re.search(r"\b(diskon|potongan)\b", seg):
            continue
        item = next((p for p in sorted(produk, key=lambda x: len(x["nama"]), reverse=True)
                     if normalisasi(p["nama"]) in seg or
                     re.search(r"(?<!\w)" + re.escape(normalisasi(p["data"].get("ID Barang", ""))) + r"(?!\w)", seg)), None)
        if item is None:
            # Nomor model unik di dalam merek juga dapat dikenali ketika
            # pengguna menyingkat 'Apple iPhone 11' menjadi 'iPhone 11'.
            brand = cocok_merek(seg, produk)
            kode = set(re.findall(r"\b[a-z]*\d+[a-z]*\b", seg))
            kandidat = [p for p in produk if normalisasi(nama_merek(p["nama"])) in brand
                        and kode.intersection(re.findall(r"\b[a-z]*\d+[a-z]*\b", normalisasi(p["nama"]))) ]
            if len(kandidat) == 1:
                item = kandidat[0]
        if item is None:
            masalah.append(seg)
            continue
        nama = normalisasi(item["nama"])
        m = re.search(r"^\s*(\d+)\s*(?:x|unit|buah|pcs)?\s+(?=[a-z])", seg)
        belakang = seg.split(nama, 1)[1] if nama in seg else ""
        sesudah = re.search(r"^\s*(?:x\s*(\d+)|(\d+)\s*(?:unit|buah|pcs))\b", belakang)
        jumlah = int(m.group(1)) if m else int(next(g for g in sesudah.groups() if g)) if sesudah else 1
        if jumlah <= 0:
            masalah.append(seg)
            continue
        kunci = item["data"].get("ID Barang") or item["nama"]
        pesanan[kunci] = (item, pesanan.get(kunci, (item, 0))[1] + jumlah)
    if masalah or not pesanan:
        return "Saya belum bisa menghitung pesanan karena model atau jumlahnya belum jelas: **" + ", ".join(masalah or segmen) + "**. Sebutkan nama model/ID dan jumlah, misalnya `2 iPhone 11 dan 1 Samsung Galaxy A15`."
    baris = []
    subtotal = 0
    stok_kurang = []
    for item, jumlah in pesanan.values():
        nilai = item["harga"] * jumlah
        subtotal += nilai
        baris.append(f"- {jumlah} × **{item['nama']}** @ {rupiah(item['harga'])} = **{rupiah(nilai)}**")
        if jumlah > item["stok"]:
            stok_kurang.append(f"{item['nama']} (diminta {jumlah}, stok {item['stok']})")
    hasil = "**Simulasi belanja berdasarkan harga katalog:**\n\n" + "\n".join(baris) + f"\n\n**Subtotal: {rupiah(subtotal)}**"
    diskon = re.search(r"\b(?:diskon|potongan)\s*(\d+(?:[.,]\d+)?)\s*%", q)
    if diskon:
        persen = float(diskon.group(1).replace(",", "."))
        if not 0 <= persen <= 100:
            return "Persentase diskon harus berada antara 0% dan 100%."
        potongan = round(subtotal * persen / 100)
        hasil += f"\nDiskon simulasi {persen:g}%: −{rupiah(potongan)}\n\n**Total simulasi: {rupiah(subtotal - potongan)}**"
        hasil += "\nDiskon ini hanya asumsi perhitungan, bukan promo toko."
    if stok_kurang:
        hasil += "\n\n**Stok belum mencukupi:** " + "; ".join(stok_kurang) + ". Pesanan belum dapat dipastikan."
    else:
        hasil += "\n\nIni simulasi, belum membuat pesanan atau mengurangi stok."
    return hasil


def jawaban_perhitungan(q, produk):
    """DIUBAH: Ranking, agregasi, anggaran, dan valuasi dihitung dari CSV."""
    order = ringkasan_pesanan(q, produk)
    if order is not None:
        return order
    disebut = cocok_merek(q, produk)
    data = [p for p in produk if normalisasi(nama_merek(p["nama"])) in disebut] if disebut else produk
    tahun = re.search(r"\b(?:tahun|rilis|buatan)\s*(20\d{2})\b", q)
    if tahun:
        data = [p for p in data if p["data"].get("Tahun Pembuatan", "") == tahun.group(1)]
    cakupan = ", ".join(sorted({nama_merek(p["nama"]) for p in data}, key=normalisasi)) if disebut else "seluruh katalog"
    if tahun:
        cakupan += f" tahun {tahun.group(1)}"
    if not data:
        return "Tidak ada produk yang cocok dengan kategori tersebut dalam CSV."

    urut_harga = bool(re.search(r"\b(termahal|tertinggi|paling mahal|paling tinggi|termurah|terendah|terrendah|paling murah|paling rendah|urutkan|daftar harga|ranking)\b", q))
    if urut_harga:
        turun = bool(re.search(r"\b(termahal|tertinggi|paling mahal|paling tinggi|descending|terbesar)\b", q))
        naik = bool(re.search(r"\b(termurah|terendah|terrendah|paling murah|paling rendah|ascending)\b", q))
        if naik and turun:
            lo, hi = min(data, key=lambda p:p["harga"]), max(data, key=lambda p:p["harga"])
            return f"Di {cakupan}, **termurah**: {lo['nama']} ({rupiah(lo['harga'])}); **termahal**: {hi['nama']} ({rupiah(hi['harga'])})."
        n = re.search(r"\b(\d{1,3})\s*(?:ponsel|model|produk|hp|harga|termahal|termurah)\b", q)
        batas = min(max(int(n.group(1)) if n else 5, 1), 100)
        urut = sorted(data, key=lambda p:(-p["harga"] if turun else p["harga"], normalisasi(p["nama"])))
        label = "tertinggi" if turun else "terendah"
        return f"**{min(batas,len(urut))} harga {label}** di {cakupan}:\n\n" + "\n".join(format_produk(p) for p in urut[:batas])

    kelompok = bool(re.search(r"\b(per|berdasarkan|tiap|setiap|masing-masing|rincian)\s+(?:merek|merk|brand|tahun|kategori|ram|penyimpanan|memori)\b", q))
    if kelompok and re.search(r"\b(stok|stock|unit|model|produk|jumlah)\b", q):
        per_tahun = bool(re.search(r"\b(?:per|berdasarkan|tiap|setiap)\s+tahun\b", q))
        per_ram = bool(re.search(r"\b(?:per|berdasarkan|tiap|setiap)\s+ram\b", q))
        per_memori = bool(re.search(r"\b(?:per|berdasarkan|tiap|setiap)\s+(?:penyimpanan|memori)\b", q))
        grup = {}
        for p in data:
            if per_tahun:
                k = p["data"].get("Tahun Pembuatan", "Tidak diketahui")
            elif per_ram or per_memori:
                label = "RAM" if per_ram else "penyimpanan"
                m = re.search(r"\b" + label + r"\s*(\d+\s*(?:GB|TB))\b", p["data"].get("Spesifikasi", ""), re.I)
                k = m.group(1).upper().replace(" ", "") if m else "Tidak diketahui"
            else:
                k = nama_merek(p["nama"])
            model, unit = grup.get(k, (0, 0))
            grup[k] = (model + 1, unit + p["stok"])
        return f"**Ringkasan stok {cakupan}:**\n\n" + "\n".join(f"- {k}: {u:,} unit ({m} model)".replace(f"{u:,}", f"{u:,}".replace(",", ".")) for k,(m,u) in sorted(grup.items(), key=lambda x:normalisasi(x[0]))) + f"\n\n**Total: {sum(p['stok'] for p in data):,} unit.**".replace(f"{sum(p['stok'] for p in data):,}", f"{sum(p['stok'] for p in data):,}".replace(",", "."))

    if re.search(r"\b(nilai|valuasi|modal)\b", q) and re.search(r"\b(stok|persediaan|inventaris)\b", q):
        nilai = sum(p["harga"] * p["stok"] for p in data)
        return f"Nilai persediaan {cakupan} berdasarkan **harga jual katalog × stok** adalah **{rupiah(nilai)}**. Ini bukan modal pembelian atau keuntungan."
    if re.search(r"\b(rata-rata|rerata|average)\b", q) and re.search(r"\b(harga|stok)\b", q):
        kol = "harga" if "harga" in q else "stok"
        rata = sum(p[kol] for p in data) / len(data)
        nilai = rupiah(round(rata)) if kol == "harga" else f"{rata:.2f}".replace(".", ",") + " unit"
        return f"Rata-rata {kol} per model di {cakupan}: **{nilai}**, dari {len(data)} model."
    if re.search(r"\b(stok kosong|habis|stok 0|stok nol)\b", q):
        kosong = [p for p in data if p["stok"] == 0]
        return f"Ada **{len(kosong)} model** dengan stok kosong di {cakupan}." + ("\n\n" + "\n".join(format_produk(p) for p in kosong[:20]) if kosong else "")
    budget = rupiah_dari_teks(q)
    if budget is not None and re.search(r"\b(di bawah|dibawah|kurang dari|maksimal|maksimum|sampai|budget|anggaran)\b", q):
        hasil = sorted([p for p in data if 0 < p["harga"] <= budget and p["stok"] > 0], key=lambda p:(p["harga"],normalisasi(p["nama"])))
        return f"Ada **{len(hasil)} model tersedia** di {cakupan} dengan harga maksimal {rupiah(budget)}." + ("\n\n" + "\n".join(format_produk(p) for p in hasil[:10]) if hasil else "") + (f"\n\nMenampilkan 10 dari {len(hasil)} model." if len(hasil)>10 else "")
    return None


def jawaban_data(pertanyaan, produk):
    """DIUBAH: Jawaban katalog dan hitungan memakai SEMUA baris CSV."""
    q = normalisasi(pertanyaan)
    if not produk:
        return None
    hasil_hitung = jawaban_perhitungan(q, produk)
    if hasil_hitung is not None:
        return hasil_hitung
    merek = {normalisasi(nama_merek(p["nama"])): nama_merek(p["nama"]) for p in produk}
    disebut = cocok_merek(q, produk)
    # DIUBAH: Daftar merek tidak pernah dialihkan ke TOP_K=3 pada RAG.
    tanya_merek = bool(re.search(r"\b(merek|merk|merekk|brand|brend)\b", q))
    tanya_daftar = bool(re.search(r"\b(apa saja|apa aja|daftar|list|punya|tersedia|ada|berapa|total|jumlah|semua|sebutkan)\b", q))
    if tanya_merek and not disebut and tanya_daftar:
        urut = sorted(merek.values(), key=normalisasi)
        if re.search(r"\b(berapa|total|jumlah)\b", q) and not re.search(r"\b(apa|daftar|list|sebutkan)\b", q):
            return f"Saat ini ada **{len(urut)} merek** dalam katalog."
        return f"Saat ini ada **{len(urut)} merek** dalam katalog:\n\n" + ", ".join(urut) + "."

    tanya_stok = bool(re.search(r"\b(stok|stock|unit|persediaan|tersedia)\b", q))
    tanya_harga = bool(re.search(r"\b(harga|price|berapa rupiah)\b", q))
    tanya_model = bool(re.search(r"\b(model|tipe|type|produk|ponsel|hp|handphone|ada|punya)\b", q))
    tanya_total = bool(re.search(r"\b(total|jumlah|keseluruhan|semua|berapa banyak)\b", q))

    # ID barang adalah kunci tepat; pertanyaan 'HP0006' tidak boleh dibaca
    # sebagai pertanyaan seluruh merek HP.
    id_produk = next((p for p in produk if p["data"].get("ID Barang", "").casefold() in re.findall(r"[\w]+", q)), None)
    if id_produk:
        p = id_produk
        detail = p["data"].get("Spesifikasi", "")
        return f"**{p['nama']}** (ID {p['data'].get('ID Barang')}) · {rupiah(p['harga'])} · stok **{p['stok']} unit**." + (f" Spesifikasi: {detail}." if detail else "")

    # DIUBAH: Nama model yang tercantum di CSV diprioritaskan atas merek.
    # 'iPhone 11' tidak lagi dijawab dengan total semua iPhone.
    nama_cocok = [p for p in produk if normalisasi(p["nama"]) in q]
    if nama_cocok:
        p = max(nama_cocok, key=lambda x: len(x["nama"]))
        detail = p["data"].get("Spesifikasi", "")
        return f"**{p['nama']}** · {rupiah(p['harga'])} · stok **{p['stok']} unit**." + (f" Spesifikasi: {detail}." if detail and not tanya_stok else "")

    # DIUBAH: Salah ketik pada merek masih bisa dipadankan dengan kode/nomor
    # model yang tepat, misalnya 'iphon 11' atau 'samsng A15'.
    kode_tanya = {x for x in re.findall(r"\b[a-z]*\d+[a-z]*\b", q)}
    if disebut and kode_tanya:
        kandidat = [p for p in produk
                    if normalisasi(nama_merek(p["nama"])) in disebut
                    and kode_tanya.intersection(re.findall(r"\b[a-z]*\d+[a-z]*\b", normalisasi(p["nama"]))) ]
        if len(kandidat) == 1:
            p = kandidat[0]
            return f"**{p['nama']}** · {rupiah(p['harga'])} · stok **{p['stok']} unit**."

    # Sebutan merek + nomor model yang tidak ada harus dijawab dengan jujur.
    if disebut and re.search(r"\b\d+[a-z]*\b", q) and (tanya_stok or tanya_harga or tanya_model):
        return "Saya belum menemukan model dengan nomor tersebut dalam katalog. Sebutkan nama model lengkap atau ID barang agar saya bisa mengecek dengan tepat."

    terpilih = [p for p in produk if normalisasi(nama_merek(p["nama"])) in disebut] if disebut else produk
    if tanya_stok and (tanya_total or not disebut or re.search(r"\bberapa\b", q)):
        cakupan = ", ".join(sorted((merek[m] for m in disebut), key=normalisasi)) if disebut else "seluruh ponsel"
        total = f"{sum(p['stok'] for p in terpilih):,}".replace(",", ".")
        return f"Total stok **{cakupan}: {total} unit**, dihitung dari {len(terpilih)} model dalam CSV."
    if disebut and (tanya_merek or tanya_model or tanya_stok or tanya_harga or tanya_daftar):
        daftar = sorted(terpilih, key=lambda p: (p["stok"] == 0, p["harga"]))
        total = f"{sum(p['stok'] for p in daftar):,}".replace(",", ".")
        baris = [format_produk(p) for p in daftar[:20]]
        tambahan = f"\n\nMenampilkan 20 dari {len(daftar)} model." if len(daftar) > 20 else ""
        return f"Ada **{len(daftar)} model** dengan total **{total} unit**:\n\n" + "\n".join(baris) + tambahan
    if tanya_total and re.search(r"\b(model|tipe|type|produk|ponsel|hp)\b", q):
        return f"Ada **{len(produk)} model** dari **{len(merek)} merek** dalam katalog."
    return None


# DIUBAH: Muat ulang data ketika isi CSV berubah, tanpa perlu menghapus cache manual.
def sidik_data(folder):
    return tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns)
                 for p in sorted(folder.glob("*.csv")) + sorted(folder.glob("*.pdf"))) if folder.is_dir() else ()


@st.cache_resource(show_spinner="Menyiapkan katalog ponsel...")
def siapkan_chatbot(sidik):
    model = buat_model()
    dokumen = muat_dokumen_produk(KNOWLEDGE_DIR)
    vectorstore = bangun_vectorstore(dokumen)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    system_prompt = muat_system_prompt(SYSTEM_PROMPT_PATH)
    return buat_rag_chain(retriever, model, system_prompt)


load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY belum diisi. Periksa file .env atau Streamlit Secrets.")
    st.stop()

try:
    produk = baca_produk(KNOWLEDGE_DIR)
    rag_chain = siapkan_chatbot(sidik_data(KNOWLEDGE_DIR))
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

if "riwayat" not in st.session_state:
    st.session_state.riwayat = []
if "pesanan_terakhir" not in st.session_state:
    st.session_state.pesanan_terakhir = None

# DIUBAH: Tema dashboard modern dengan kontras yang nyaman pada desktop dan ponsel.
st.markdown("""<style>
.stApp {background: radial-gradient(circle at 80% 0%, #e5eeff 0%, #f7f9fd 38%, #f7f9fd 100%); color:#17243b;}
.block-container {max-width:1120px; padding-top:2rem;}
.hero {background:linear-gradient(115deg,#101e39,#263e71 65%,#3863a4); color:white;
       border-radius:24px;padding:32px 36px;box-shadow:0 18px 42px rgba(28,54,105,.18);}
.hero .eyebrow {font-size:.77rem;letter-spacing:.17em;text-transform:uppercase;color:#a9c5fa;font-weight:700;}
.hero h1 {color:white;font-size:clamp(2rem,4vw,3.15rem);margin:.3rem 0 .5rem;line-height:1.1;}
.hero p {color:#dbe6fb;font-size:1rem;margin:0;max-width:650px;}
[data-testid="stMetric"] {background:white;border:1px solid #e3eaf5;border-radius:18px;
 padding:16px 20px;box-shadow:0 8px 24px rgba(25,45,84,.045);}
[data-testid="stMetricValue"] {color:#17376a;}
[data-testid="stSidebar"] {background:#f1f5fc;}
[data-testid="stChatMessage"] {border-radius:17px;border:1px solid #e5ebf4;background:rgba(255,255,255,.9);}
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📱 Ponsel Studio")
    st.caption("Asisten pilihan ponsel dari katalog toko")
    st.divider()
    st.markdown("**Contoh pertanyaan**")
    st.markdown("• 5 harga tertinggi dan terendah  \n• Total stok per merek / per tahun  \n• Beli 2 iPhone 11 dan 3 Samsung Galaxy A15  \n• Ponsel maksimal Rp3 juta")
    st.divider()
    if st.button("↻ Mulai percakapan baru", use_container_width=True):
        st.session_state.riwayat = []
        st.session_state.pesanan_terakhir = None
        st.rerun()
    st.caption("Harga dan stok merujuk pada CSV di knowledge_docs.")

st.markdown("""<div class="hero"><div class="eyebrow">Katalog multi merek · Asisten belanja</div>
<h1>Temukan ponsel yang pas.</h1><p>Jelajahi pilihan, bandingkan spesifikasi, dan cek stok dengan cepat.</p></div>""",
            unsafe_allow_html=True)
st.write("")
kolom = st.columns(3)
kolom[0].metric("Model tersedia", f"{len(produk):,}".replace(",", "."))
kolom[1].metric("Total unit stok", f"{sum(p['stok'] for p in produk):,}".replace(",", "."))
kolom[2].metric("Merek", len({normalisasi(nama_merek(p["nama"])) for p in produk}))
st.caption("Ringkasan dan simulasi belanja dihitung dari CSV. Harga belum termasuk biaya lain; simulasi tidak membuat pesanan.")
st.markdown("#### Tanya asisten produk")

if not st.session_state.riwayat:
    with st.chat_message("assistant"):
        st.markdown("Halo! Saya bisa bantu cek pilihan, harga, stok, dan menghitung simulasi belanja. Misalnya, **'Berapa total stok per merek?'** atau **'Beli 2 iPhone 11 dan 3 Samsung Galaxy A15'**.")
for pesan in st.session_state.riwayat:
    with st.chat_message(pesan["role"]):
        st.markdown(pesan["isi"])

pertanyaan = st.chat_input("Tanyakan harga, stok, perbandingan, atau total belanja...")
if pertanyaan:
    with st.chat_message("user"):
        st.markdown(pertanyaan)
    st.session_state.riwayat.append({"role": "user", "isi": pertanyaan})
    with st.chat_message("assistant"):
        # DIUBAH: Pertanyaan lanjutan sederhana memakai pesanan yang terakhir
        # dihitung dalam sesi ini, tanpa menyimpan atau membuat order sungguhan.
        lanjut_total = bool(re.fullmatch(r"\s*(?:jadi\s+)?(?:berapa\s+)?(?:total(?:nya)?|subtotal(?:nya)?|jumlah(?:nya)?)(?:\s+berapa)?\s*[?.!]*", normalisasi(pertanyaan)))
        if lanjut_total and st.session_state.pesanan_terakhir:
            jawaban = jawaban_data(st.session_state.pesanan_terakhir, produk)
        else:
            jawaban = jawaban_data(pertanyaan, produk)
            if re.search(r"\b(beli|pesan|order|checkout|belanja|pembelanjaan|keranjang)\b", normalisasi(pertanyaan)) and jawaban and "**Subtotal:" in jawaban:
                st.session_state.pesanan_terakhir = pertanyaan
        if jawaban is None:
            with st.spinner("Menyiapkan rekomendasi..."):
                jawaban = st.write_stream(rag_chain.stream(pertanyaan))
        else:
            st.markdown(jawaban)
    st.session_state.riwayat.append({"role": "assistant", "isi": jawaban})
