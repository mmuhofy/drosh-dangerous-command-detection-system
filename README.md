# Drosh — Tehlikeli Komut Algılama Sistemi

Drosh (`github.com/mmuhofy/Drosh`, paket `dev.drosh`) için **tehlikeli komut
algılama modeli**nin eğitim ve değerlendirme hattı.

Kullanıcı komutu yazar → model değerlendirir → model zararlı bulursa ekranda
uyarı çıkar, kullanıcı swipe ile kapatır. Bu repo modelin kendisini üretir.

> **Durum: Faz 1 tamamlandı, model henüz eğitilmedi.**
> Normalizasyon sözleşmesi (Python ↔ JavaScript) kuruldu ve kilitlendi.
> Prototip şu an sahte bir skor gösteriyor; gerçek model Faz 3-4'te geliyor.
> Bkz. [`docs/COMMAND-RISK-MODEL.md`](docs/COMMAND-RISK-MODEL.md).

---

## Ne üretiyor

| Artefakt | Ne işe yarar | Git'te |
|---|---|---|
| `ml/src/drosh_ml/normalize.py` | **Normalizasyon sözleşmesinin kaynağı.** Komutun modele nasıl girdiğini belirler | ✅ |
| `html/command-risk/risk-scoring.js` | Sözleşmenin JavaScript kopyası | ✅ |
| `html/command-risk/index.html` | Tarayıcı prototipi — `file://` üzerinden çift tıklayıp açılır | ✅ |
| `ml/artifacts/model.joblib` | Python tarafı model kalıcılığı | ❌ |
| `ml/data/seed_from_muhofy.txt` | **Elle doldurulur** — gerçek komutların | ✅ |
| `html/command-risk/risk-model.js` | Eğitilmiş model, sıfır bağımlılık | ⏳ Faz 4 |
| `html/command-risk/golden-vectors.json` | Python ↔ JS skor parite kanıtı | ⏳ Faz 4 |

## Neden tarayıcıda çalışıyor

Model doğrusal: bir n-gram sözlüğü ve bir float dizisi. Skor, basitçe

```
skor = bias + Σ (ağırlık[i] × sözlük[i'de var mı])
```

Bu JavaScript'te birkaç satır. Bu yüzden prototipte **WASM yok, ONNX Runtime
yok, npm yok, build step yok**. `index.html` dosyasına çift tıkla, çalışır.

Aynı nedenle `joblib` bu projenin uygulama tarafında hiçbir rol oynamaz —
`.joblib` bir pickle dosyasıdır, JS'te veya Kotlin'te yüklenemez. `export.py`
modeli düz float dizisi olarak yeniden ifade eder.

## Pipeline

```
ml/data/seed_from_muhofy.txt   senin gerçek komutların
        │
        ├─ grammar.py          yıkıcı komut grameri (etiket yapıdan doğru)
        ├─ negatives.py        "tehlikeli görünüp masum" komutlar
        └─ augment.py          obfuscation varyantları
        │
        ▼
   dataset (40-60k satır)  →  ordinal 3 sınıf: safe / risky / destructive
        │
        ▼
   train.py       doğrusal model, presence-temelli char n-gram (2..5)
        │
        ├─ evaluate.py     kategori bazlı precision/recall + confusion matrix
        │                 + TUTULMUŞ obfuscation ailesi skoru
        ▼
   export.py      → risk-model.js  (VOCAB + katsayılar + eşikler)
                 → golden-vectors.json (300 komutun Python skorları)
        │
        ▼
   index.html     tarayıcıda, sunucusuz
```

## Hızlı başlangıç

### Codespaces (önerilen)

```bash
# GitHub'da Code → Codespaces → Create codespace on drosh-dangerous-command-detection-system
# devcontainer otomatik bootstrap olur
scripts/bootstrap.sh
```

### Yerel

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ml pytest
```

## Komutlar

```bash
scripts/bootstrap.sh                    # ortamı kur
scripts/train.sh                        # dataset → train → evaluate → export
scripts/pin.sh                          # requirements.lock'u dondur
scripts/sync-js.sh                      # Python tablosunu JS'e yeniden yaz

python -m drosh_ml.normalize            # normalizasyon sözleşmesi self-test
node html/command-risk/risk-scoring-selftest.mjs   # JS tarafı self-test
python -m pytest ml/tests -v            # çapraz dil parite testleri
```

## En kritik iki kural

**1. Python ve JavaScript normalizasyonu birebir aynı olmak zorunda.**
Eğitim Python'da yapılıyor, çalışma zamanı JavaScript'te. Aralarında tek bir
fark olsa, değerlendirilen model ile çalışan model farklı model olur ve hiçbir
offline metrik bunu göstermez. Garantiyi üç şey sağlıyor:

- `normalize.py` Unicode katlama tablosunun tek sahibidir; `sync-js.sh` onu
  JS'e üretir, elle düzenleme yok.
- `ml/tests/test_parity.py` 82 girdi üzerinde iki uygulamayı bayt bayt
  karşılaştırır — Türkçe karakterler, birleşik işaretler, görünmez karakterler,
  ANSI kaçışları, çok segmentli zincirler dahil.
- Faz 4'te `golden-vectors.json` 300 komutun **tam float skorlarını** Python'da
  kaydedip tarayıcıda yeniden hesaplar; sapma 1e-9'u aşarsa prototip kırmızıya
  döner.

**2. Normalizasyon çıktısı her zaman saf ASCII.**
0x20–0x7E dışındaki her şey atılır. Böylece Python kod noktası, JS UTF-16 kod
birimı ve bayt aynı şey olur — `ğ` bir yerde 1 birim, diğerinde yarım birim
olsaydı n-gram dilimlemesi sessizce kayardı.

## Veri lisansı

Tamamı sentetik veya sentetik türevidir; elde yazılmış bir veri kaynağı yoktur.
Bu, Drosh ile aynı GPL-3.0 lisansı altında sorunsuz bir şekilde birleşmesini
sağlar.

## Katkı

`ml/data/seed_from_muhofy.txt` dosyasına gerçek komutların eklenmesi bu projenin
en değerli katkısı. Dosyanın başındaki yönergeleri okuyun — özellikle
**"tehlikeli görünüp masum olan komutlar"** bölümü, modelin en çok zorlandığı
sınıf.

## Lisans

GPL-3.0-or-later — Drosh ile aynı lisans. Bkz. [LICENSE](LICENSE).