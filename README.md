YouTube içerik asistanı uygulaması. İşte bu proje için detaylı bir kapsam dosyası ve çalıştırma talimatları:

---

## Proje Kapsam Dosyası: YouTube İçerik Asistanı

**1. Proje Adı:**  
YouTube İçerik Asistanı

**2. Proje Amacı:**  
Bu proje, kullanıcılara YouTube videolarından ses indirme, indirilen sesleri yüksek doğrulukla metne dönüştürme ve sonrasında bu metin içeriği hakkında yerel bir Büyük Dil Modeli (LLM) ve RAG (Retrieval-Augmented Generation) sistemi kullanarak sorular sorma imkanı sunan, uçtan uca bir Streamlit web uygulaması geliştirmeyi amaçlamaktadır.

**3. Ana Modüller ve Özellikler:**

```
  ``**A. `youtube_downloader.py` (YouTube Ses İndirici):**     *   Kullanıcının girdiği YouTube URL'sinden videonun sesini indirme.     *   `yt-dlp` kütüphanesini kullanma ve gerekirse otomatik yükleme.     *   Video başlığını alıp dosya sistemi için güvenli bir dosya adı oluşturma (özel karakterleri temizleme, boşlukları `_` ile değiştirme, uzunluğu kısaltma).     *   MP3 formatında ve 192kbps kalitede ses indirme.     *   İndirilen ses dosyalarını `./audios` klasörüne kaydetme.     *   FFmpeg için yaygın Windows yollarını kontrol etme ve `yt-dlp`'ye bildirme.  **B. `audio_transcriber.py` (Sesli Metne Dönüştürücü):**     *   İndirilen veya kullanıcı tarafından yüklenen ses dosyalarını metne dönüştürme.     *   `faster-whisper` kütüphanesini kullanma ve gerekirse otomatik yükleme.     *   Farklı Whisper model boyutları (tiny, base, small, medium, large-v2) seçeneği sunma.     *   GPU varsa CUDA ile, yoksa CPU ile işlem yapma.     *   Model dosyaları için geçici ve güvenli indirme yolları ayarlama.     *   Metinleri segmentler halinde döndürme.     *   Oluşturulan transkriptleri `./transcripts` klasörüne zaman damgalı olarak kaydetme.     *   Geçici işlem dosyaları için klasör oluşturma ve işlem sonrası temizleme.  **C. `rag_helper.py` (RAG Yardımcı Modülü):**     *   **Paket Kurulumu:** RAG için gerekli `sentence-transformers`, `faiss-cpu`, `langchain`, `llama-cpp-python` gibi paketlerin kontrolü ve gerekirse kurulumu.     *   **Model İndirme:**         *   `all-MiniLM-L6-v2` gömme (embedding) modelini (`sentence-transformers` ile) `./models/embedding_model` altına indirme.         *   LLM modeli (`llama-2-7b-chat.Q4_K_M.gguf` önerilir) için `./models` klasöründe varlık kontrolü ve kullanıcıya indirme talimatı verme.     *   **RAGProcessor Sınıfı:**         *   Gömme modelini yükleme.         *   `RecursiveCharacterTextSplitter` (Langchain) ile metni anlamlı parçalara (chunks) bölme.         *   Parçaların gömmelerini oluşturma ve FAISS ile bir vektör indeksi oluşturma.         *   Oluşturulan FAISS indeksini (`.index`) ve metin parçalarını (`.chunks.json`) `./rag_indexes` klasörüne kaydetme ve geri yükleme.         *   Bir sorgu (soru) için en alakalı metin parçalarını FAISS indeksinden çekme.     *   **LocalLLM Sınıfı:**         *   `llama-cpp-python` kullanarak yerel GGUF formatındaki LLM'i (örn: Llama-2-7B-Chat) yükleme.         *   GPU varsa katmanları GPU'ya offload etme denemesi.         *   Soru ve RAG ile çekilen alakalı metin parçalarını kullanarak LLM'e bir prompt oluşturma ve cevap üretme. **D. `app.py` (Ana Streamlit Uygulaması):**     *   **Kullanıcı Arayüzü:**         *   Sekmeli (tabs) arayüz: "Ses İndir", "Metne Dönüştür", "RAG Hazırla", "Soru Sor", "Dosyalar".         *   Özelleştirilmiş CSS ile modern ve kullanıcı dostu bir görünüm.         *   Formlar, butonlar, dosya yükleyiciler, metin giriş alanları, seçiciler.         *   İşlem durumlarını göstermek için `st.status` ve `st.spinner`.         *   Başarı ve hata mesajları.     *   **İş Akışı:**         1.  **Ses İndirme:** YouTube URL'si ile sesi indirir.         2.  **Metne Dönüştürme:** Son indirilen sesi veya yüklenen bir ses dosyasını seçilen Whisper modeli ile metne dönüştürür. Sonucu gösterir ve indirilebilir metin dosyası olarak sunar.         3.  **RAG Hazırlama:**             *   Gerekli RAG ve LLM paket/modellerini kurma/hazırlama butonları.             *   Son oluşturulan veya seçilen bir transkript dosyasını RAG için işler (parçalama, gömme, FAISS indeksi oluşturma ve kaydetme).         4.  **Soru Sorma:**             *   Hazırlanmış bir RAG indeksini seçme/yükleme.             *   LLM yüklenmemişse kullanıcıyı uyarma.             *   Kullanıcının video içeriği hakkında soru sormasına izin verme.             *   Soruyu RAG sistemi ile işleyip alakalı parçaları bulma.             *   Alakalı parçalar ve soru ile LLM'e prompt gönderip cevap alma.             *   Sohbet geçmişini (soru-cevap) gösterme.     *   **Dosya Yönetimi:**         *   `./audios`, `./transcripts`, `./models`, `./rag_indexes` klasörlerini oluşturma.         *   "Dosyalar" sekmesinde bu klasörlerdeki mevcut dosyaları listeleme (bu kısım kodda eksik, ama mantıksal bir eklenti olabilir).     *   **Session State Yönetimi:** Sohbet geçmişi, yüklenen LLM, RAG işlemcisi, güncel transkript/indeks yolları gibi durumları oturum boyunca saklama.``
```

**4. Kullanılan Teknolojiler:**

- **Programlama Dili:** Python
  
- **Web Framework:** Streamlit
  
- **Ses İndirme:** yt-dlp
  
- **Ses Transkripsiyonu:** faster-whisper, torch
  
- **Metin İşleme ve RAG:**
  
  - sentence-transformers (gömme için)
    
  - faiss-cpu (vektör veritabanı)
    
  - langchain (metin bölücü için)
    
- **Yerel LLM Çalıştırma:** llama-cpp-python
  
- **Veri Saklama/Serileştirme:** json (chunks için)
  
- **Diğer:** pandas (requirements'ta var ama kodda doğrudan kullanılmıyor gibi), os, sys, subprocess, time, re, unicodedata, shutil
  

**5. Hedef Kitle:**

- YouTube içeriklerinden bilgi çıkarmak, özetlemek veya soru sormak isteyen kullanıcılar.
  
- İçerik üreticileri, öğrenciler, araştırmacılar.
  
- Yerel LLM ve RAG sistemleriyle pratik uygulamalar geliştirmek isteyenler.
  

**6. Ön Koşullar ve Kısıtlamalar:**

- **FFmpeg:** yt-dlp'nin MP3'e dönüştürme yapabilmesi için sistemde FFmpeg'in kurulu ve PATH'e ekli olması şiddetle tavsiye edilir (özellikle Windows'ta). youtube_downloader.py bazı yaygın yolları kontrol etmeye çalışır.
  
- **Sistem Kaynakları:**
  
  - Whisper modelleri (özellikle medium ve large-v2) ve LLM'ler (örn: Llama-2 7B) önemli miktarda RAM ve CPU/GPU kaynağı gerektirir.
    
  - GPU (NVIDIA CUDA destekli) varsa performans önemli ölçüde artar.
    
- **Model İndirme Süreleri:** Gömme modeli, Whisper modelleri ve özellikle LLM dosyaları büyük olabilir ve ilk indirmeleri zaman alabilir.
  
- **İnternet Bağlantısı:** Paketlerin ve modellerin indirilmesi için gereklidir.
  
- rag_helper.py şu anda faiss-cpu kullanmaktadır. GPU varsa faiss-gpu ve ilgili CUDA toolkit kurulumu ile performans artırılabilir.
  
- LLM olarak llama-2-7b-chat.Q4_K_M.gguf önerilmiştir, ancak farklı GGUF modelleri de kullanılabilir (kodda model adı değiştirilerek).
  

---

## Nasıl Çalıştırılır?

Bu kapsamlı Streamlit uygulamasını yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

**1. Dosyaların Hazırlanması:**

- Bir proje klasörü oluşturun (örneğin, youtube_content_assistant).
  
- Aşağıdaki beş dosyayı bu klasörün içine kaydedin:
  
  - app.py
    
  - youtube_downloader.py
    
  - audio_transcriber.py
    
  - rag_helper.py
    
  - requirements.txt
    
- Proje klasörünüzde aşağıdaki alt klasörleri oluşturun (uygulama bunları kendi de oluşturur ama önceden yapmakta fayda var):
  
  - audios
    
  - transcripts
    
  - models
    
  - rag_indexes
    

**2. Sanal Ortam Oluşturma ve Aktifleştirme (Şiddetle Önerilir):**  
Bir sanal ortam kullanmak, projenizin bağımlılıklarını sistem genelindeki Python kurulumunuzdan izole eder ve olası kütüphane çakışmalarını önler.

- Proje klasörünüzün (youtube_content_assistant) içindeyken terminali veya komut istemcisini açın.
  
- Sanal ortamı oluşturun:
  
  ```
    `python -m venv venv`
  ```
  

- Sanal ortamı aktifleştirin:
  
  - **Windows için:**
    
    ```
      `.\venv\Scripts\activate`
    ```
    

- Aktifleştirme başarılı olduğunda, komut satırınızın başında (venv) gibi bir ifade görmelisiniz.
  

Aktifleştirme başarılı olduğunda, komut satırınızın başında (venv) gibi bir ifade görmelisiniz.

**3. Gerekli Kütüphanelerin Kurulumu:**  
Sanal ortam aktifken, requirements.txt dosyasında listelenen temel kütüphaneleri kurun. Diğer özel kütüphaneler (yt-dlp, faster-whisper, RAG için olanlar) uygulama içinden gerektiğinde kurulmaya çalışılacaktır, ancak temel olanları baştan kurmak iyi bir adımdır.

```
  `pip install -r requirements.txt`
```

Not: torch kurulumu bazen sisteminize (CPU/GPU, CUDA sürümü) göre özelleştirme gerektirebilir. Eğer pip install torch sorun çıkarırsa, PyTorch resmi sitesinden ([pytorch.org](https://www.google.com/url?sa=E&q=https%3A%2F%2Fpytorch.org%2F)) sisteminize uygun komutu alarak kurun.

**4. FFmpeg Kurulumu (Önemli):**  
yt-dlp'nin indirdiği sesi MP3'e düzgün çevirebilmesi için FFmpeg gereklidir.

- FFmpeg'i [ffmpeg.org](https://www.google.com/url?sa=E&q=https%3A%2F%2Fffmpeg.org%2Fdownload.html) adresinden indirin.
  
- Arşivden çıkan bin klasörünün içindeki ffmpeg.exe, ffprobe.exe (ve ffplay.exe) dosyalarını sisteminizin PATH'ine ekli bir yere (örn: C:\ffmpeg\bin oluşturup buraya kopyalayın ve bu yolu PATH'e ekleyin) veya youtube_downloader.py dosyasının aradığı yaygın konumlardan birine koyun.
  
- Terminalde ffmpeg -version komutu çalışıyorsa kurulum başarılıdır.
  

**5. LLM Modelinin İndirilmesi:**  
Uygulamanın soru-cevap özelliği için bir GGUF formatında LLM'e ihtiyacınız olacak.

- rag_helper.py içinde llama-2-7b-chat.Q4_K_M.gguf modeli önerilmektedir.
  
- Bu modeli [Hugging Face - TheBloke/Llama-2-7B-Chat-GGUF](https://www.google.com/url?sa=E&q=https%3A%2F%2Fhuggingface.co%2FTheBloke%2FLlama-2-7B-Chat-GGUF%2Ftree%2Fmain) adresinden indirin.
  
- İndirdiğiniz .gguf dosyasını projenizdeki ./models/ klasörünün içine kopyalayın. (Dosya adı tam olarak llama-2-7b-chat.Q4_K_M.gguf olmalı veya rag_helper.py içindeki LocalLLM sınıfında model adını güncellemelisiniz).
  

**6. Uygulamanın Çalıştırılması:**  
Tüm önkoşullar tamamlandıktan ve sanal ortamınız aktifken, Streamlit uygulamasını başlatmak için proje klasörünüzün içindeyken terminalde şu komutu çalıştırın:

```
  `streamlit run app.py`
```

Bu komut, Streamlit uygulamasını başlatacak ve varsayılan web tarayıcınızda genellikle http://localhost:8501 adresinde açacaktır.

**7. Uygulama İçi Kurulum Adımları (İlk Kullanımda):**  
Uygulama açıldığında, çeşitli sekmelerdeki butonları kullanarak kalan bağımlılıkları (yt-dlp, faster-whisper, RAG paketleri) ve modelleri (gömme modeli, Whisper modeli) indirebilir/kurabilirsiniz.

- **"Ses İndir" Sekmesi:** YouTube URL'si girip "İndir" butonuna bastığınızda yt-dlp kontrol edilecek ve gerekirse yüklenecektir.
  
- **"Metne Dönüştür" Sekmesi:** "Metne Dönüştür" butonuna bastığınızda faster-whisper ve seçtiğiniz Whisper modeli kontrol edilecek/yüklenecektir.
  
- **"RAG Hazırla" Sekmesi:**
  
  - "RAG Bağımlılıklarını Kur" ile sentence-transformers, faiss-cpu, langchain gibi paketler kurulur.
    
  - "LLM Modelini Hazırla" ile gömme modeli (all-MiniLM-L6-v2) indirilir ve ./models/ içindeki LLM varlığı kontrol edilir.
    
  - Bir transkript seçip "RAG İçin Hazırla" dediğinizde RAG indeksi oluşturulur.
    
- **"Soru Sor" Sekmesi:** Burada LLM'i yükleyip sorularınızı sorabilirsiniz.
  

**8. Sanal Ortamdan Çıkış (İşiniz Bittiğinde):**  
Uygulamayla işiniz bittiğinde ve terminali kapatmadan önce sanal ortamı devre dışı bırakmak için:

```
  `deactivate`
```

Bu komut sizi sisteminizin genel Python ortamına döndürecektir.

Bu adımlar, YouTube İçerik Asistanı uygulamanızı başarıyla çalıştırmanıza yardımcı olacaktır. Özellikle model indirme ve ilk kurulumlar biraz zaman alabilir, sabırlı olun.
