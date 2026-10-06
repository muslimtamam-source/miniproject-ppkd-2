"""
RAG Chatbot sederhana pakai LangChain + Groq + ChromaDB (lokal)

Script ini adalah versi .py dari materi yang sudah dibedah di sesi LangChain:
- Blok 1: Model, Prompt Template, Output Parser
- Blok 2: LCEL (pipe, RunnableParallel, RunnablePassthrough)
- Blok 3: Data Ingestion (Document, Chunking, Embedding, Vector Store)
- Blok 4: RAG Chain (Context Injection, chain final)

Knowledge document di sini pakai 3 artikel berita asli soal RUU Pelindungan
Ketenagakerjaan (September 2026), disimpan sebagai file .pdf di folder
knowledge_docs/.

Loader PDF-nya pakai PyMuPDF4LLMLoader dari package langchain-pymupdf4llm,
package resmi terpisah (bukan dari langchain_community yang sudah sunset).

Cara jalanin:
    python rag_chatbot.py

Prasyarat:
    - File .env berisi GROQ_API_KEY di folder yang sama dengan script ini
    - Folder knowledge_docs/ berisi file .pdf yang mau dijadikan sumber
    - Package sudah terinstall (lihat requirements di bagian bawah file)
"""

import os
import glob

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.documents import Document
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ============================================================
# 1. KONFIGURASI
# ============================================================

CHAT_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "ruu_ketenagakerjaan"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

# Folder berisi knowledge document. Semua file .pdf di dalamnya akan
# dianggap sebagai satu sumber pengetahuan terpisah.
KNOWLEDGE_DIR = "./knowledge_docs"

# System prompt disimpan terpisah dari kode, supaya bisa diubah/di-review
# tanpa menyentuh logika program, dan supaya jejak revisinya jelas kalau
# dipakai bersama version control (git).
SYSTEM_PROMPT_PATH = "./system_prompt.md"


# ============================================================
# 2. SETUP MODEL DAN EMBEDDING
# ============================================================

def buat_model() -> ChatGroq:
    """Siapkan koneksi ke model chat lewat Groq."""
    return ChatGroq(
        model=CHAT_MODEL,
        temperature=0,
        reasoning_effort="low",
    )


def buat_embeddings() -> HuggingFaceEmbeddings:
    """Siapkan model embedding, jalan lokal tanpa API key tambahan."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# ============================================================
# 3. DATA INGESTION (Load -> Split -> Embed -> Store)
# ============================================================

def muat_dokumen(folder: str) -> list[Document]:
    """
    Load semua file .pdf di dalam folder jadi list of Document, pakai
    PyMuPDF4LLMLoader langsung dari library LangChain (langchain-pymupdf4llm).

    Setiap file di-load lewat loader resmi ini, jadi kita tidak perlu
    menulis logika parsing PDF sendiri, cukup panggil .load() untuk
    masing-masing file lalu digabung jadi satu list.
    """
    daftar_dokumen = []
    path_file = sorted(glob.glob(os.path.join(folder, "*.pdf")))
    # glob.glob(...) bertugas mencari file yang sudah ditentukan di os.path.join(), 
    # dan hasilnya berupa daftar nama file yang cocok, dalam bentuk list Python.

    if not path_file:
        raise FileNotFoundError(
            f"Tidak ada file .pdf ditemukan di folder '{folder}'. "
            "Pastikan folder knowledge_docs/ berisi file sumber."
        )

    for path in path_file:
        # mode="single" -> satu file PDF jadi satu Document (bukan per halaman).
        # use_layout=False -> ekstraksi teks polos, tanpa mesin deteksi layout/
        # OCR yang tidak perlu untuk PDF berbasis teks seperti artikel ini
        # (mempercepat proses dan menghindari pesan "Using Tesseract..." di
        # konsol yang bisa bikin peserta bingung).
        loader = PyMuPDF4LLMLoader(file_path=path, mode="single", use_layout=False)
        daftar_dokumen.extend(loader.load())

    return daftar_dokumen


def bangun_vectorstore(dokumen: list, embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Ubah kumpulan Document jadi vector store siap dicari.

    Tahapannya persis yang sudah dibedah di Blok 3:
    1. Load   -> sudah dilakukan di muat_dokumen()
    2. Split  -> potong jadi chunk kecil
    3. Embed + Store -> simpan ke ChromaDB
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    potongan = splitter.split_documents(dokumen)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    # reset_collection() dipakai supaya aman dijalankan berulang kali
    # (script ini bisa dijalankan ulang tanpa bikin data dobel atau error).
    vectorstore.reset_collection()
    vectorstore.add_documents(potongan)

    return vectorstore


# ============================================================
# 4. RAG CHAIN (Context Injection -> Prompt -> Model -> Parser)
# ============================================================

def format_docs(daftar_dokumen: list[Document]) -> str:
    """Gabungkan beberapa chunk hasil retrieval jadi satu teks konteks."""
    return "\n\n".join(dok.page_content for dok in daftar_dokumen)

def muat_system_prompt(path: str) -> str:
    """
    Baca system prompt dari file .md terpisah.

    Dipisah dari kode supaya system prompt bisa direvisi (oleh siapa pun
    yang bertanggung jawab atas kualitas jawaban chatbot) tanpa perlu
    menyentuh atau memahami kode Python-nya sama sekali.
    """
    with open(path, encoding="utf-8") as f:
        return f.read()

def buat_rag_chain(retriever, model: ChatGroq, system_prompt: str):
    """
    Satukan retriever, prompt, model, dan parser jadi satu chain LCEL.

    system_prompt (dari file system_prompt.md) berisi instruksi statis:
    peran asisten dan aturan anti-halusinasi. Placeholder {context} dan
    {question} sengaja TIDAK ditaruh di file itu, karena dua-duanya
    bukan instruksi tetap, melainkan bagian yang diisi ulang setiap kali
    ada pertanyaan baru, jadi tetap disusun di sini sebagai pesan "human".
    """
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Konteks:\n{context}\n\nPertanyaan: {question}"),
    ])

    rag_chain = (
        RunnableParallel(
            context=retriever | format_docs,
            question=RunnablePassthrough(),
        )
        | rag_prompt
        | model
        | StrOutputParser()
    )
    return rag_chain


# ============================================================
# 5. PROGRAM UTAMA
# ============================================================

def main():
    # load_dotenv() harus dipanggil sebelum ChatGroq dibuat, supaya
    # GROQ_API_KEY sudah ada di environment variable saat dibutuhkan.
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY tidak ditemukan. Pastikan file .env ada di folder "
            "yang sama dengan script ini dan berisi GROQ_API_KEY=..."
        )

    print("Menyiapkan model dan embedding...")
    model = buat_model()
    embeddings = buat_embeddings()

    print(f"Memuat dokumen dari folder '{KNOWLEDGE_DIR}'...")
    dokumen = muat_dokumen(KNOWLEDGE_DIR)
    print(f"  -> {len(dokumen)} dokumen berhasil dimuat.")

    print("Membangun vector store dari dokumen...")
    vectorstore = bangun_vectorstore(dokumen, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    print(f"Memuat system prompt dari '{SYSTEM_PROMPT_PATH}'...")
    system_prompt = muat_system_prompt(SYSTEM_PROMPT_PATH)
    
    print("Merakit RAG chain...")
    rag_chain = buat_rag_chain(retriever, model, system_prompt)

    print("\nRAG chatbot siap. Ketik pertanyaan, atau 'keluar' untuk berhenti.\n")

    while True:
        pertanyaan = input("Pertanyaan: ").strip()
        if pertanyaan.lower() in {"keluar", "exit", "quit"}:
            print("Sampai jumpa.")
            break
        if not pertanyaan:
            continue

        jawaban = rag_chain.invoke(pertanyaan)
        print(f"Jawaban : {jawaban}\n")


if __name__ == "__main__":
    main()

# ============================================================
# KENAPA ADA "if __name__ == '__main__':" DI SINI?
# ============================================================
#
# Python punya "label" tersembunyi bernama __name__ di setiap file .py.
# Isinya beda tergantung cara file ini dipakai:
#
#   - Kalau file ini dijalankan LANGSUNG (python rag_chatbot.py),
#     maka __name__ otomatis berisi teks "__main__".
#
#   - Kalau file ini "dipinjam" fungsinya oleh file lain, misalnya
#     lewat "from rag_chatbot import muat_dokumen", maka __name__
#     berisi nama file ini sendiri, yaitu "rag_chatbot", BUKAN "__main__".
#
# Baris "if __name__ == '__main__':" ini jadi semacam pertanyaan yang
# ditanyakan Python ke dirinya sendiri: "Apakah saya sedang dijalankan
# langsung, atau cuma dipinjam file lain?"
#
# Kalau jawabannya "dijalankan langsung", baru main() dipanggil, dan
# seluruh program (load dokumen, bikin vectorstore, tanya-jawab) jalan.
#
# Kalau jawabannya "cuma dipinjam", main() TIDAK dipanggil. Ini penting
# supaya kalau suatu saat ada script lain yang cuma mau meminjam satu
# fungsi kecil dari sini (misalnya muat_dokumen() saja), seluruh RAG
# chatbot ini tidak ikut menyala tanpa diminta.
#
# Letaknya WAJIB di paling bawah file, setelah semua fungsi (def ...)
# selesai didefinisikan. Python membaca file dari atas ke bawah, jadi
# kalau baris ini ditaruh sebelum fungsi-fungsi yang dipanggil di dalam
# main() selesai "dikenalkan" ke Python, akan muncul error NameError.
# ============================================================
