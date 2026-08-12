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

**Faz 1 — Proje iskeleti.** Şu an yalnızca proje yapısı, temel FastAPI
uygulaması ve `/health` endpoint'i mevcut. Doküman yükleme, chunking,
embedding, Qdrant entegrasyonu ve LLM entegrasyonu henüz eklenmedi.

## Gereksinimler

- Python 3.11
- Docker + Docker Compose (Qdrant için)

## Kurulum (geliştirme)

```bash
python -m venv .venv
.venv/Scripts/activate  # Windows
pip install -e ".[dev]"
cp .env.example .env
```

## Çalıştırma

```bash
uvicorn app.main:app --reload
```

## Test

```bash
pytest
ruff check .
mypy
```

## Qdrant (Docker Compose)

```bash
docker compose up -d qdrant
```

## Not

Bu depoda model ağırlıkları, PDF dokümanları, Qdrant verisi, checkpoint'ler
veya gerçek API anahtarları **bulunmaz**. `.env` dosyası commit edilmez;
yalnızca `.env.example` referans olarak eklenir.
