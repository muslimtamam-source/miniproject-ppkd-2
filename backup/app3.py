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


def jawaban_data(pertanyaan, produk):
    """DIUBAH: Jawaban katalog dan hitungan memakai SEMUA baris CSV."""
    q = normalisasi(pertanyaan)
    if not produk:
        return None
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
    st.markdown("• Ada iPhone apa saja?  \n• Berapa total stok semua ponsel?  \n• Rekomendasi ponsel kamera bagus")
    st.divider()
    if st.button("↻ Mulai percakapan baru", use_container_width=True):
        st.session_state.riwayat = []
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
st.caption("Ringkasan dihitung langsung dari seluruh baris CSV. Jika PDF juga ada, isinya dipakai untuk pertanyaan umum.")
st.markdown("#### Tanya asisten produk")

if not st.session_state.riwayat:
    with st.chat_message("assistant"):
        st.markdown("Halo! Saya siap membantu Anda mencari ponsel. Coba tanyakan **'Ada IPhone apa saja?'** atau **'Berapa total stok semua ponsel?'**")
for pesan in st.session_state.riwayat:
    with st.chat_message(pesan["role"]):
        st.markdown(pesan["isi"])

pertanyaan = st.chat_input("Tanyakan merek, harga, spesifikasi, atau total stok...")
if pertanyaan:
    with st.chat_message("user"):
        st.markdown(pertanyaan)
    st.session_state.riwayat.append({"role": "user", "isi": pertanyaan})
    with st.chat_message("assistant"):
        jawaban = jawaban_data(pertanyaan, produk)
        if jawaban is None:
            with st.spinner("Menyiapkan rekomendasi..."):
                jawaban = st.write_stream(rag_chain.stream(pertanyaan))
        else:
            st.markdown(jawaban)
    st.session_state.riwayat.append({"role": "assistant", "isi": jawaban})
