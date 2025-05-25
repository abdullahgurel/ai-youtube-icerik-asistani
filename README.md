# YouTube İçerik Asistanı

Bu uygulama, YouTube videolarından ses indirip, indirilen sesi metne dönüştürmenize ve içerikle ilgili sorular sormanıza olanak tanır. Yerel bir LLM modeli kullanarak video içeriği hakkında sorular cevaplayabilirsiniz.

## Kurulum

### 1. Gerekli Python Paketleri

Uygulamayı çalıştırmak için aşağıdaki paketleri kurmanız gerekir:

```bash
pip install streamlit pandas yt-dlp faster-whisper sentence-transformers faiss-cpu langchain llama-cpp-python
```

### 2. LLM Modelini İndirme

Uygulama, soru-cevap özelliği için yerel bir LLM modeli kullanır. Aşağıdaki adımları izleyerek modeli indirebilirsiniz:

1. https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/tree/main adresine gidin
2. `llama-2-7b-chat.Q4_K_M.gguf` dosyasını indirin (veya daha küçük bir model seçebilirsiniz)
3. İndirdiğiniz dosyayı `./models/` dizinine kopyalayın

### 3. FFmpeg Kurulumu (Windows için)

YouTube'dan ses indirmek için FFmpeg gereklidir. Windows kullanıcıları için:

1. https://ffmpeg.org/download.html adresinden FFmpeg'i indirin
2. İndirdiğiniz zip dosyasını bir klasöre çıkarın (örn. `C:\ffmpeg`)
3. Çıkardığınız dosyaların içindeki `bin` klasörünü sistem PATH'ine ekleyin:
   - Sistem özellikleri > Gelişmiş > Ortam Değişkenleri > Path > Düzenle
   - `C:\ffmpeg\bin` gibi bir yol ekleyin

## Uygulamayı Çalıştırma

Uygulamayı çalıştırmak için aşağıdaki komutu kullanın:

```bash
streamlit run app.py
```

## Kullanım

Uygulama dört ana adımdan oluşur:

1. **Ses İndir**: YouTube video URL'sini girerek sesi indirin
2. **Metne Dönüştür**: İndirilen sesi metne dönüştürün
3. **RAG Hazırla**: Metni RAG sistemi için hazırlayın ve LLM modelini yükleyin
4. **Soru Sor**: Video içeriği hakkında sorular sorun

## Sık Karşılaşılan Sorunlar

### GPU Kullanımı

LLM modelinin daha hızlı çalışması için CUDA destekli bir GPU kullanabilirsiniz. Bunun için:

```bash
pip uninstall llama-cpp-python -y
pip install llama-cpp-python --extra-index-url=https://llama-cpp-python.netlify.app/whl/cu118
```

### Bellek Sorunları

LLM modeli çok büyükse ve bilgisayarınızda yetersiz bellek varsa, daha küçük bir model kullanabilirsiniz (örneğin `llama-2-7b-chat.Q2_K.gguf`).

## Önemli Not

Bu uygulama eğitim ve kişisel kullanım amaçlıdır. Telif hakkı korumalı içerikleri indirirken yasal sorumluluklar size aittir.