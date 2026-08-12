# Tokenizer Yaklaşımı

Chunking servisi, `app/services/tokenization.py` içindeki `Tokenizer` protokolü
arkasında çalışır. Bu, chunking mantığının hangi tokenizer'ın kullanıldığından
bağımsız olmasını sağlar.

## Şu an kullanılan: `ApproximateTokenizer`

- **Yöntem:** Unicode kelime/noktalama ayrıştırması (`\w+|[^\w\s]` regex'i).
  Her kelime ve her noktalama işareti bir "token" sayılır.
- **Neden bu yaklaşım seçildi:** Faz 3'te henüz bir embedding modeli
  seçilmedi (donanım/uyumluluk kontrolü tamamlanmadı). Gerçek bir model
  tokenizer'ı (ör. BGE-M3, multilingual-E5 için HuggingFace tokenizer'ı veya
  tiktoken/BPE) hem ek bağımlılık hem de bazı durumlarda internetten indirme
  gerektirir; bu da "mümkün olduğunca yerel" hedefiyle çelişir. Dependency-free,
  tamamen deterministik ve offline çalışan bir yaklaşım tercih edildi.
- **Sınırlamalar:** Bu, hiçbir gerçek embedding/LLM modelinin subword
  vocabulary'sini birebir yansıtmaz. Token sayıları yönsel olarak doğrudur
  (daha uzun metin → daha çok token) ama herhangi bir gerçek modelin tokenizer
  çıktısıyla birebir eşleşmeyecektir. Bu bir tasarım tercihidir, ölçülmüş bir
  performans iddiası değildir (bkz. proje kuralı 16).

## Değiştirilebilirlik

`ChunkingService`, `Tokenizer` protokolüne (`count_tokens`, `token_boundaries`)
uyan herhangi bir nesneyi kabul eder. Faz 4'te embedding modeli seçildiğinde,
o modelin gerçek tokenizer'ını sarmalayan bir `Tokenizer` implementasyonu
(ör. HuggingFace `AutoTokenizer` tabanlı) enjekte edilerek chunking mantığında
hiçbir değişiklik yapılmadan geçiş yapılabilir.

## Chunking konfigürasyonları

`app/services/chunking_presets.py` içinde isimle seçilebilir üç deney
konfigürasyonu tanımlıdır: `300_50`, `500_75`, `800_100` (max_tokens/overlap_tokens).
Bu isimler benchmark script'lerinden dışarıdan seçilebilir olacak şekilde
tasarlanmıştır.
