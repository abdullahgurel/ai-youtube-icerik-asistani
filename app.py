"""
YouTube Ses İndirici, Sesli Metne Dönüştürücü ve Soru Cevaplama Uygulaması
Bu uygulama, YouTube videolarından ses indirme, 
indirilen sesi metne dönüştürme ve içerikle ilgili soru sorma işlevlerini sunar.
"""

import streamlit as st
import os
import sys
import time
import pandas as pd
from datetime import datetime
import json

# YouTube modülünü içe aktar
sys.path.append(".")
from youtube_downloader import sanitize_filename, install_yt_dlp, download_youtube_audio
from audio_transcriber import install_packages as install_whisper_packages, transcribe_audio
from rag_helper import install_packages as install_rag_packages, download_model_if_needed, RAGProcessor, LocalLLM

# Sayfa yapılandırması
st.set_page_config(
    page_title="YouTube İçerik Asistanı",
    page_icon="🎵",
    layout="wide"
)

# CSS stilleri
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }

    .status-area {
        background-color: #2c2c2c;  /* koyu gri arka plan */
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        color: #ffffff;  /* beyaz yazı */
    }

    .success {
        color: #81c784;  /* açık yeşil */
        font-weight: bold;
    }

    .error {
        color: #e57373;  /* açık kırmızı */
        font-weight: bold;
    }

    .stButton>button {
        width: 100%;
        background-color: #424242;  /* koyu buton */
        color: #ffffff;
        border: none;
        border-radius: 0.25rem;
    }

    .stButton>button:hover {
        background-color: #616161;  /* hover efekti */
    }

    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        color: #ffffff;
    }

    .user-message {
        background-color: #3a3a3a;  /* koyu pembe yerine koyu gri */
        border-left: 5px solid #64b5f6;  /* mavi kenar çizgisi */
    }

    .assistant-message {
        background-color: #4a4a4a;
        border-left: 5px solid #81c784;  /* yeşil kenar çizgisi */
    }
    </style>
""", unsafe_allow_html=True)

# Başlık ve açıklama
st.title("🎵📝🤖 YouTube İçerik Asistanı")
st.markdown("""
    Bu uygulama, YouTube videolarından ses indirip, indirilen sesi metne dönüştürmenize ve 
    içerikle ilgili sorular sormanıza olanak tanır. Dört adımdan oluşur:
    
    1. YouTube URL'sini girin ve sesi indirin
    2. İndirilen sesi metne dönüştürün
    3. Metni RAG sistemi ile analiz edin
    4. Video içeriği hakkında sorular sorun
""")

# Ana dizinleri oluştur
if not os.path.exists("./audios"):
    os.makedirs("./audios")
if not os.path.exists("./transcripts"):
    os.makedirs("./transcripts")
if not os.path.exists("./models"):
    os.makedirs("./models")
if not os.path.exists("./rag_indexes"):
    os.makedirs("./rag_indexes")

# Sekmeleri oluştur
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Ses İndir", "Metne Dönüştür", "RAG Hazırla", "Soru Sor", "Dosyalar"])

# Session state başlangıç değerleri
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "rag_processor" not in st.session_state:
    st.session_state["rag_processor"] = None
if "llm" not in st.session_state:
    st.session_state["llm"] = None
if "llm_loaded" not in st.session_state:
    st.session_state["llm_loaded"] = False
if "current_transcript_path" not in st.session_state:
    st.session_state["current_transcript_path"] = None
if "current_transcript_title" not in st.session_state:
    st.session_state["current_transcript_title"] = None

# Ses indirme sekmesi
with tab1:
    st.header("YouTube'dan Ses İndir")
    
    # Form oluştur
    with st.form(key="download_form"):
        youtube_url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
        col1, col2 = st.columns(2)
        
        with col1:
            output_dir = st.text_input("Çıktı Dizini", value="./audios")
        
        download_button = st.form_submit_button("🔽 İndir")
        
    # İndirme işlemini gerçekleştir
    if download_button and youtube_url:
        with st.status("İndirme işlemi başlatılıyor...") as status:
            try:
                st.write("YouTube URL'si doğrulanıyor...")
                
                # yt-dlp'yi yükle
                if not install_yt_dlp():
                    st.error("yt-dlp yüklenemedi!")
                    status.update(label="İndirme başarısız", state="error")
                else:
                    st.write("yt-dlp kurulumu başarılı!")
                    
                    # Ses indirme işlemi
                    st.write("Ses indiriliyor...")
                    audio_path = download_youtube_audio(youtube_url, output_dir)
                    
                    # Başarılı
                    st.session_state["last_downloaded"] = audio_path
                    status.update(label="İndirme başarılı!", state="complete")
                    
                    st.success(f"Ses başarıyla indirildi: {audio_path}")
                    st.audio(audio_path)
                    
            except Exception as e:
                st.error(f"İndirme sırasında bir hata oluştu: {str(e)}")
                status.update(label="İndirme başarısız", state="error")

# Metne dönüştürme sekmesi
with tab2:
    st.header("Ses Dosyasını Metne Dönüştür")
    
    # Form oluştur
    with st.form(key="transcribe_form"):
        # Dosya yükleme veya son indirilen dosyayı kullanma seçeneği
        use_last_downloaded = st.checkbox("Son indirilen dosyayı kullan", 
                                         value="last_downloaded" in st.session_state)
        
        if use_last_downloaded and "last_downloaded" in st.session_state:
            audio_file_path = st.session_state["last_downloaded"]
            st.info(f"Kullanılacak dosya: {audio_file_path}")
        else:
            audio_file_path = st.file_uploader("Ses Dosyası Seç", type=["mp3", "wav", "m4a", "ogg"])
            if audio_file_path:
                # Yüklenen dosyayı geçici bir dosyaya kaydet
                bytes_data = audio_file_path.read()
                temp_file = f"./audios/temp_{int(time.time())}_{audio_file_path.name}"
                with open(temp_file, "wb") as f:
                    f.write(bytes_data)
                audio_file_path = temp_file
                st.session_state["last_uploaded"] = audio_file_path
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_size = st.selectbox(
                "Model Boyutu", 
                ["tiny", "base", "small", "medium", "large-v2"],
                index=3  # Varsayılan olarak "medium" seçili
            )
            
        transcribe_button = st.form_submit_button("🔊 Metne Dönüştür")
    
    # Transcribe işlemini gerçekleştir
    if transcribe_button and (audio_file_path is not None):
        with st.status("Metne dönüştürme işlemi başlatılıyor...") as status:
            try:
                # Paketleri yükle
                st.write("Gerekli paketler kontrol ediliyor...")
                if install_whisper_packages():
                    st.write("Gerekli paketler kuruldu.")
                    
                    # Metne dönüştürme işlemi
                    st.write(f"{model_size} modeli yükleniyor... (Bu biraz zaman alabilir)")
                    transcript_segments = transcribe_audio(audio_file_path, model_size)
                    
                    # Tam metni oluştur
                    full_transcript = "\n".join(transcript_segments)
                    
                    # Metni göster
                    st.subheader("Dönüştürülen Metin:")
                    st.markdown(full_transcript)
                    
                    # Metni dosyaya kaydet
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    base_filename = os.path.basename(audio_file_path).rsplit(".", 1)[0]
                    transcript_file = f"./transcripts/{base_filename}_{timestamp}.txt"
                    
                    with open(transcript_file, "w", encoding="utf-8") as f:
                        f.write(full_transcript)
                    
                    # Session state'e kaydet
                    st.session_state["current_transcript_path"] = transcript_file
                    st.session_state["current_transcript_title"] = base_filename
                    
                    # Metni indirme butonu
                    st.download_button(
                        label="📝 Metni İndir",
                        data=full_transcript,
                        file_name=f"{base_filename}_{timestamp}.txt",
                        mime="text/plain"
                    )
                    
                    status.update(label="Dönüştürme başarılı!", state="complete")
                    
                else:
                    st.error("Gerekli paketler yüklenemedi!")
                    status.update(label="Dönüştürme başarısız", state="error")
                    
            except Exception as e:
                st.error(f"Metne dönüştürme sırasında bir hata oluştu: {str(e)}")
                status.update(label="Dönüştürme başarısız", state="error")

# RAG Hazırlama sekmesi
with tab3:
    st.header("RAG Sistemi Hazırla")
    
    # RAG için gerekli paketleri yükle
    if st.button("🔧 RAG Bağımlılıklarını Kur"):
        with st.status("RAG için gerekli paketler yükleniyor...") as status:
            if install_rag_packages():
                st.success("RAG paketleri başarıyla kuruldu!")
                status.update(label="Kurulum başarılı", state="complete")
            else:
                st.error("RAG paketleri kurulumunda hata!")
                status.update(label="Kurulum başarısız", state="error")
    
    # LLM modelini hazırla
    if st.button("🧠 LLM Modelini Hazırla"):
        with st.status("LLM modeli kontrol ediliyor...") as status:
            if download_model_if_needed():
                st.success("LLM modeli hazır!")
                status.update(label="Model hazır", state="complete")
                
                # LLM'i yükle
                if "llm" not in st.session_state or st.session_state["llm"] is None:
                    st.session_state["llm"] = LocalLLM()
                
                with st.spinner("LLM modeli yükleniyor... (Bu işlem birkaç dakika sürebilir)"):
                    if st.session_state["llm"].load_model():
                        st.session_state["llm_loaded"] = True
                        st.success("LLM modeli başarıyla yüklendi!")
                    else:
                        st.error("LLM modeli yüklenemedi!")
            else:
                st.error("LLM modeli bulunamadı!")
                st.markdown("""
                    LLM modelini indirmek için:
                    1. https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/tree/main adresine gidin
                    2. llama-2-7b-chat.Q4_K_M.gguf dosyasını indirin
                    3. İndirdiğiniz dosyayı `./models/` dizinine kopyalayın
                """)
                status.update(label="Model bulunamadı", state="error")
    
    # Transcript seçimi ve RAG hazırlama
    st.subheader("Transcript'i RAG İçin Hazırla")
    
    # Transcript dosyalarını listele
    transcript_files = []
    if os.path.exists("./transcripts"):
        transcript_files = [f for f in os.listdir("./transcripts") if f.endswith(".txt")]
    
    # Transcript seçimi
    selected_transcript = None
    
    # Son oluşturulan transcript'i kullan
    use_current_transcript = st.checkbox(
        "Son dönüştürülen metni kullan", 
        value="current_transcript_path" in st.session_state and st.session_state["current_transcript_path"] is not None
    )
    
    if use_current_transcript and "current_transcript_path" in st.session_state and st.session_state["current_transcript_path"]:
        selected_transcript = st.session_state["current_transcript_path"]
        st.info(f"Kullanılacak transcript: {os.path.basename(selected_transcript)}")
    else:
        if transcript_files:
            transcript_option = st.selectbox(
                "Transcript dosyası seç", 
                options=transcript_files,
                format_func=lambda x: x
            )
            if transcript_option:
                selected_transcript = os.path.join("./transcripts", transcript_option)
        else:
            st.warning("Henüz transcript dosyası bulunmuyor. Önce bir ses dosyasını metne dönüştürün.")
    
    # RAG hazırlama butonu
    if selected_transcript and st.button("🔍 RAG İçin Hazırla"):
        with st.status("Transcript RAG için hazırlanıyor...") as status:
            try:
                # Transcript dosyasını oku
                with open(selected_transcript, "r", encoding="utf-8") as f:
                    transcript_text = f.read()
                
                # RAG işleyicisini oluştur
                if "rag_processor" not in st.session_state or st.session_state["rag_processor"] is None:
                    st.session_state["rag_processor"] = RAGProcessor()
                
                # Transcripti işle
                if st.session_state["rag_processor"].process_transcript(transcript_text):
                    # RAG indeksini kaydet
                    base_name = os.path.basename(selected_transcript).rsplit(".", 1)[0]
                    index_path = f"./rag_indexes/{base_name}"
                    
                    if st.session_state["rag_processor"].save_index(index_path):
                        st.success(f"RAG indeksi başarıyla oluşturuldu ve kaydedildi: {index_path}")
                        st.session_state["current_rag_index"] = index_path
                        status.update(label="RAG hazırlama başarılı", state="complete")
                    else:
                        st.error("RAG indeksi kaydedilemedi!")
                        status.update(label="RAG indeksi kaydedilemedi", state="error")
                else:
                    st.error("Transcript RAG için hazırlanamadı!")
                    status.update(label="RAG hazırlama başarısız", state="error")
                    
            except Exception as e:
                st.error(f"RAG hazırlama sırasında bir hata oluştu: {str(e)}")
                status.update(label="RAG hazırlama başarısız", state="error")

# Soru Sorma sekmesi
with tab4:
    st.header("Video İçeriği Hakkında Soru Sor")
    
    # RAG indekslerini listele
    rag_indexes = []
    if os.path.exists("./rag_indexes"):
        # .index uzantılı dosyaları bul ve .index uzantısını kaldır
        rag_indexes = [f.rsplit(".", 1)[0] for f in os.listdir("./rag_indexes") if f.endswith(".index")]
    
    # Eğer RAG indeksi varsa
    if rag_indexes:
        # Son oluşturulan RAG indeksini kullan
        use_current_index = st.checkbox(
            "Son oluşturulan RAG indeksini kullan", 
            value="current_rag_index" in st.session_state
        )
        
        selected_index = None
        if use_current_index and "current_rag_index" in st.session_state:
            selected_index = st.session_state["current_rag_index"]
            index_basename = os.path.basename(selected_index)
            st.info(f"Kullanılacak RAG indeksi: {index_basename}")
        else:
            index_option = st.selectbox(
                "RAG indeksi seç", 
                options=[os.path.basename(idx) for idx in rag_indexes],
                format_func=lambda x: x
            )
            if index_option:
                selected_index = os.path.join("./rag_indexes", index_option)
        
        # Eğer LLM yüklenmemişse uyarı ver
        if not st.session_state.get("llm_loaded", False):
            st.warning("LLM henüz yüklenmedi. Soru sormadan önce 'RAG Hazırla' sekmesinden LLM modelini yükleyin.")
        
        # Eğer indeks seçildi ve LLM yüklendiyse
        if selected_index and st.session_state.get("llm_loaded", False):
            # RAG indeksini yükle
            if ("rag_processor" not in st.session_state or 
                st.session_state["rag_processor"] is None or
                not hasattr(st.session_state["rag_processor"], "index") or
                st.session_state["rag_processor"].index is None):
                
                with st.spinner("RAG indeksi yükleniyor..."):
                    st.session_state["rag_processor"] = RAGProcessor()
                    if st.session_state["rag_processor"].load_index(selected_index):
                        st.success("RAG indeksi başarıyla yüklendi")
                    else:
                        st.error("RAG indeksi yüklenemedi!")
            
            # Sohbet geçmişi göster
            if st.session_state["chat_history"]:
                st.subheader("Sohbet Geçmişi")
            for message in st.session_state["chat_history"]:
                    if message["role"] == "user":
                        st.markdown(f"""
                        <div class="chat-message user-message">
                            <b>Soru:</b> {message["content"]}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-message assistant-message">
                            <b>Cevap:</b> {message["content"]}
                        </div>
                        """, unsafe_allow_html=True)
            
            # Yeni soru sorma alanı
            st.subheader("Yeni Soru")
            with st.form(key="question_form"):
                question = st.text_area("İçerik hakkında bir soru sorun", height=100)
                submit_button = st.form_submit_button("🔍 Soru Sor")
                
                if submit_button and question:
                    # Kullanıcı sorusunu kaydet
                    st.session_state["chat_history"].append({
                        "role": "user",
                        "content": question
                    })
                    
                    # İlgili chunk'ları getir
                    with st.spinner("İlgili içerik aranıyor..."):
                        relevant_chunks = st.session_state["rag_processor"].retrieve_relevant_chunks(
                            question, top_k=3
                        )
                    
                    # LLM ile cevap oluştur
                    with st.spinner("Cevap oluşturuluyor..."):
                        answer = st.session_state["llm"].generate_response(
                            question, relevant_chunks
                        )
                    
                    # Cevabı kaydet
                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": answer
                    })
                    
                    # Sayfayı yenile (son eklenen mesajları göstermek için)
                    st.rerun()
        else:
            st.warning("Soru sormadan önce LLM modelini yükleyin ve RAG indeksini hazırlayın.")
    else:
        st.warning("Henüz hiç RAG indeksi oluşturulmamış. Önce 'RAG Hazırla' sekmesinden bir indeks oluşturun.")