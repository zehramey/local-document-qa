# local-document-qa

Yerel (mümkün olduğunca offline) çalışan bir PDF/TXT doküman soru-cevap sistemi.

## Amaç

Kullanıcının yüklediği PDF/TXT dokümanları parçalara ayırıp yerel bir embedding
modeliyle vektörleştirmek, bu vektörleri Qdrant'ta saklamak, kullanıcı sorusuna
en alakalı parçaları (context) bulup yerel bir LLM'e göndermek ve üretilen
cevabı kaynak dosya/sayfa/chunk bilgisiyle birlikte sunmaktır. Doküman
içeriğinde olmayan bilgiler uydurulmaz (grounded / cevap yoksa "bilmiyorum").

## Mimari (üst seviye)

- `app/api` — FastAPI endpoint'leri
- `app/core` — konfigürasyon, ortak altyapı
- `app/domain` — domain modelleri / iş kuralları
- `app/models` — Pydantic veri şemaları
- `app/repositories` — Qdrant gibi depolama katmanlarına erişim
- `app/services` — chunking, embedding, retrieval, generation servisleri
- `app/llm` — yerel LLM istemcileri
- `ui` — Streamlit arayüzü
- `tests/unit`, `tests/integration`, `tests/evaluation` — test katmanları
- `benchmark` — gerçek ölçüm/karşılaştırma scriptleri (iddia değil, ölçüm)
- `finetuning` — yalnızca baseline RAG + eval seti tamamlandıktan sonra kullanılacak

## Geliştirme Durumu

**Faz 7 — FastAPI + Streamlit arayüzü.** Doküman yükleme/listeleme/silme,
chunking, gerçek embedding (BAAI/bge-m3), Qdrant indeksleme, dense
retrieval (+ opsiyonel reranker), yerel LLM (LM Studio) ile RAG cevabı ve
kaynak doğrulama uçtan uca çalışıyor; FastAPI endpoint'leri ve Streamlit
arayüzü üzerinden kullanılabiliyor.

## Gereksinimler

- Python 3.11
- Docker + Docker Compose (Qdrant, ve isteğe bağlı olarak API/UI konteynerleri için)
- Yerel bir LLM sunucusu — bu projede [LM Studio](https://lmstudio.ai/) ile
  test edildi (OpenAI-uyumlu API, `http://localhost:1234`). Ollama veya
  başka bir OpenAI-uyumlu sunucu da `LLM_BASE_URL`/`LLM_MODEL_ID` ile
  kullanılabilir.
- (Opsiyonel, gerçek embedding/reranker modelleri için) `pip install -e ".[embeddings]"`
  — `sentence-transformers`/`torch` indirir, ilk çalıştırmada ayrıca
  BAAI/bge-m3 modeli (~2.2GB) HuggingFace'ten indirilir.

## Kurulum — Manuel (geliştirme)

```bash
python -m venv .venv
.venv/Scripts/activate  # Windows
pip install -e ".[dev,embeddings,ui]"
cp .env.example .env
docker compose up -d qdrant   # sadece Qdrant'ı Docker'da çalıştır
```

LM Studio'yu (veya başka bir yerel LLM sunucusunu) ayrıca başlatıp bir
model yükleyin; `.env` içindeki `LLM_BASE_URL`/`LLM_MODEL_ID` bu sunucuyla
eşleşmeli.

API'yi çalıştır:

```bash
uvicorn app.main:app --reload
```

Streamlit arayüzünü (ayrı bir terminalde) çalıştır:

```bash
streamlit run ui/streamlit_app.py
```

API varsayılan olarak `http://localhost:8000`, arayüz `http://localhost:8501`
adresinde açılır. Arayüz, API'nin adresini `API_BASE_URL` ortam
değişkeninden okur (varsayılan `http://localhost:8000`).

## Kurulum — Docker Compose

```bash
docker compose up -d --build
```

Bu, Qdrant + API + Streamlit UI konteynerlerinin tümünü başlatır. API
konteyneri gerçek embedding modelini (`sentence-transformers`/`torch`)
içerir; ilk çalıştırmada BAAI/bge-m3 modeli konteyner içine indirilir ve
`huggingface_cache` volume'unda saklanır (yeniden başlatmalarda tekrar
indirilmez).

**LLM sunucusu konumu:** Bu projede kullanılan LLM sunucusu (LM Studio)
konteynerize edilmedi — host makinede çalışır. `docker-compose.yml`
içinde API konteynerine `LLM_BASE_URL=http://host.docker.internal:1234`
enjekte edilir (host'taki LM Studio'ya ulaşmak için). LLM'i kendi container'ınızda
çalıştırmak isterseniz, `LLM_BASE_URL` ortam değişkenini o servisin adresine
göre değiştirmeniz yeterli — kod tarafında host/container ayrımı
tamamen konfigürasyona bağlıdır, kod değişikliği gerekmez.

## Test

```bash
pytest                    # unit testler (gerçek model indirmez)
pytest -m integration -s  # gerçek modellerle (indirme + çalışan LM Studio gerektirir)
ruff check .
mypy
```

## Qdrant'a Doğrudan Erişim (opsiyonel)

```bash
docker compose up -d qdrant
```

## Güvenlik Notları

- Dosya boyutu sınırı ve MIME/uzantı doğrulaması `FileValidator` ile
  yapılır (`MAX_UPLOAD_SIZE_BYTES`).
- Yüklenen dosyalar hiçbir zaman kullanıcı girdisinden türetilmiş bir
  dosya sistemi yoluna yazılmaz (tamamen bellek içinde işlenir); dosya adı
  ayrıca `sanitize_filename` ile path-traversal denemelerine karşı
  temizlenir.
- Tüm Qdrant sorguları `document_id` filtresiyle sınırlıdır; bir dokümana
  ait retrieval/soru-cevap işlemi başka bir dokümanın verisine erişemez.
- API hataları her zaman `{error_code, message}` şeklinde temiz bir JSON
  olarak döner; Python stack trace'i hiçbir zaman kullanıcıya gösterilmez.

## Not

Bu depoda model ağırlıkları, PDF dokümanları, Qdrant verisi, checkpoint'ler
veya gerçek API anahtarları **bulunmaz**. `.env` dosyası commit edilmez;
yalnızca `.env.example` referans olarak eklenir.
