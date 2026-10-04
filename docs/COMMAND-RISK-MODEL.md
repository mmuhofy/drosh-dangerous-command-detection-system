# Tehlikeli Komut Algılama Modeli — Teknik Doküman

_Yazılmaya başlandı: 2026-10-04_

> **Durum: Faz 1 tamamlandı. Model eğitilmedi.**
> Bu doküman normalizasyon sözleşmesiyle ilgili her şeyi kesinleştirir
> (Python ↔ JavaScript birebir aynı olmalı) ve model tasarımının gerekçesini
> kaydeder. Eğitim sonuçları, eşik değerleri ve export formatı boş
> bölümlere eklenecek.

## 1. Kapsam

Drosh'ta kullanıcı komutu yazar → model değerlendirir → zararlıysa ekranda
uyarı çıkar → kullanıcı swipe ile kapatır.

Bu repo **sadece modeli üretir**. Arayüz, uyarı kartı, kapı noktalarına
bağlama ve Drosh içindeki entegrasyon bu kapsamın dışında.

## 2. Neden bu repo ayrı

Android reposunun içine koymadık çünkü:

- scikit-learn / numpy bağımlılıkları Gradle ağacının ağırlığını artırır ve
  F-Droid derlemesine taşınacak bir yük bindirir.
- Eğitim yeniden üretilebilir olmalı — veri ve scriptler versiyonlanabilir,
  incelenebilir olmalı.
- Drosh GPL-3.0, bu repo da GPL-3.0. Model + dataset'in lisans uyumu tartışmasız.

## 3. Etiket şeması

Üç ordinal sınıf:

| Değer | Sınıf | Anlam |
|---|---|---|
| 0 | `safe` | Rutin komut, uyarı yok |
| 1 | `risky` | Çok şey yıkıyor ama günlük hayatta olağan — `git clean -fdx`, `rm -rf node_modules` |
| 2 | `destructive` | Geri dönüşü yok — `rm -rf /`, `dd if=`, fork bomb |

Çıkış (export) **ikili** karar verir: `WARN_THRESHOLD` ve `BLOCK_THRESHOLD`.
Yani bugünkü ürün davranışı tam olarak "uyarı çıkar / çıkmaz".

### Neden ikili değil

1. **Eşik başlığı, yeniden eğitim olmadan.** `risky` bölgesi `safe` ile
   `destructive` arasında gerçekten farklı bir alan: `git clean -fdx`,
   `docker system prune`, `rm -rf node_modules` burada. Model bu sınırı
   öğrenmeye zorlanınca, eşik daha sonra kaydırılabilir bir dış sabite dönüşür.
2. **Dürüst değerlendirme.** İkili model tek sayı raporlar; üçlü model *hangi
   sınırın* başarısız olduğunu raporlar. `destructive` recall'ı %97 ise, o %3'ün
   yanlış etiketli pozitif mi yoksa gerçekten belirsiz komut mu olduğunu
   bilmemiz gerekir.

### Yanlış pozitif/negatif asimetrisi

Rutin bir temizliği tehlikeli sanmak **yanlış pozitiftir**. Kullanıcı uyarıyı
görür, gereksiz bulur, swipe ile kapatır — ve bundan sonra uyarıları okumaz.
Özellik kendini ilk günden öldürür.

Tehlikeli bir komutu kaçırmak **yanlış negatiftir** ve bu veri kaybı olayıdır.

Bu yüzden eşikler `labels.py: FALSE_NEGATIVE_COST = 20.0` oranıyla seçilir,
asla 0.5'te değil. Bu bir istatistiksel değil **ürün kararıdır**; telemetriyle
ayarlanabilmesi için burada kayıt altına alınmıştır.

## 4. Model mimarisi

### 4.1 Aile A — char n-gram, presence tabanlı

```
TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=3,
                binary=True, use_idf=True, norm=None)
```

`binary=True` ve `norm=None` seçimleri keyfi değildir, matematiksel zorunluluktur:
bu ikisi olmadan **IDF'yi katsayılara katlamak (folding) mümkün değildir**.
Bu formda

```
Σ w_j · x_j  =  Σ (w_j · idf_j) · 1[j ∈ X]
```

olduğundan eğitilmiş model, `idf` dizisi taşımadan **tek düz float dizisi**
olarak export edilebilir.

**Presence (n-gram sayısı değil, var/yok) neden:**

- `rm -rf / rm -rf / rm -rf /` ile `rm -rf /` aynı puanı almalı. Sayım
  kullanılsaydı puan süperlineer büyürdü ve modeli okuyamayan biri tekrar ederek
  puanı keyfi şişirebilirdi.
- Uzunluk normalizasyonu gerekmediğinden yukarıdaki katlama birebir geçerli.
- Uzun zincirli komutların "daha tehlikeli" olması gerekiyorsa bu sinyal zaten
  dense özelliklerde var; n-gram'de olması gerekmiyor.

### 4.2 Aile B — dense sinyaller

~24 elle tanımlı sinyal (`sudo` var mı, `rm`+recursive+force üçlüsü, hedef
sistem yolu mu, `dd`, `mkfs`, `> /dev/sd`, `chmod 777`, fork bomb `:(){`,
`curl|sh`, `history -c`, `kill -9 -1`, `shutdown`, zincir segment sayısı,
`$(` derinliği, base64, `eval`, `chown -R`, `git push --force`,
`docker system prune -a`, apt purge, `truncate` …).

Bu aile iki işe yarar: skoru düzeltmek ve uyarı kartının **"bunu neden tehlikeli
sayıyor"** metnini beslemek.

### 4.3 Skor

```
skor = sigmoid(w_A · presence + w_B · z + b)
```

Tek model, iki eşik. Sınıflar ordinal regresyonla (termodinamik soft-label)
öğretilir; export ikili karar verir.

## 5. Dataset

Tamamı sentetik. ~40-60k satır, dört kaynak:

| Kaynak | Etiket nasıl belirleniyor | Neden |
|---|---|---|
| `grammar.py` | **Yapı tarafından doğru** — binary × flag × hedef yol kombinasyonları | Hacim LLM'in veremeyeceği kadar büyük olur; etiket gürültüsü sıfır |
| `negatives.py` | Kural tabanlı (elle) | Modeli kıran sınıf: tehlikeli kelime içeren ama masum komutlar |
| `seed_from_muhofy.txt` | Senin etiketin veya benim notering | Uydurma dağılımın en büyük açığı |
| `augment.py` | Etiket korunur | Obfuscation'a dayanıklılık |

### Obfuscation holdout

Obfuscation ailelerinin bir kısmı **eğitimden tamamen çıkarılıp** teste ayrılır
(`OBFUSCATION_HOLDOUT`). Sentetik veride asıl tehlike modelin semantik yerine
yüzey kalıpları öğrenmesidir; bu split tam olarak onu ölçer. "Gramerin hiç
görmediği bir obfuscation ailesini doğru sınıflandırıyor mu" sorusunun cevabı
buradan gelir.

## 6. Normalizasyon sözleşmesi

`ml/src/drosh_ml/normalize.py` — 6 adımlı boru hattı:

1. **ANSI kaçışlarını temizle.** Kontrol karakterlerinden *önce* yapılmalı,
   yoksa `ESC[31m` metne `31m` olarak sızar.
2. **Segment böl.** `[;|&]` ve yeni satırlar. Çok satırlı yapıştırmada tek
   tehlikeli satırın görünmesi için.
3. **Segment başına:** whitespace kontrolleri → boşluk, kalan kontroller
   silinir, ASCII'ye katlanır, küçük harfe çevrilir, whitespace sıkıştırılır.
4. **Yeniden birleştir:** `' ; '`. Ayraç belirgin olsun ki segment sınırını
   geçen n-gram'lar komut içi n-gram'lardan farklı olsun.
5. **İki görünüm:** `text` (tırnaklar korunur) ve `unquoted`.
6. **N-gram:** 2..5, presence.

İkinci görünüm (`unquoted`) alıntı obfuscation'ı için: `r"m" -rf /` ile
`rm -rf /` aynı n-gram kümesini vermeli. Doğrulanmıştır: %100 örtüşme.

### Çıktı her zaman saf ASCII

0x20–0x7E dışındaki her şey atılır. Bunun sonucu: Python kod noktası, JS UTF-16
kod birimi ve bayt aynı birim olur. Bu olmadan `ğ` (Python'da 1 kod noktası,
JS'te 1 yarım kod birimi) sonraki tüm n-gram konumlarını sessizce kaydırırdı.

### Katlama tablosu tek kaynaktan

`normalize.py`'deki `_FOLD_GROUPS` tablosu tek sahibidir. `scripts/sync-js.sh`
bunu JS'e **üretir**. CI'da senkronizasyon ayrıca denetlenir.

Türkçe için özel dikkat: `ş Ş ğ Ğ ı İ` NFKD ayrışmasına sahip **değildir**,
yani otomatik katlanmazlar. `İ` (U+0130) ayrışır (`I` + birleşik nokta), `ı`
(U+0131) ayrışmaz. Tablo 218 kod noktası içerir.

## 7. Parite kanıtı

| Mekanizma | Ne garantiler |
|---|---|
| `sync-js.sh` | Katlama tablosu elle kayamaz |
| `test_parity.py` | 82 girdide `text`, `unquoted`, `segments`, **n-gram kümesi** ve düşürülen karakter sayısı bayt bayt aynı |
| `golden-vectors.json` (Faz 4) | 300 komutun tam float skorları Python'dan; tarayıcıda yeniden hesaplanır, sapma < 1e-9 |

Üçüncüsü en güçlüsüdür çünkü sadece normalizasyonu değil **nihai skoru**
karşılaştırır — katsayı katlama hatası gibi sorunları da yakalar.

**Mevcut durum:** `test_parity.py` 82/82 geçiyor (0 sapma).

## 8. Export formatı

⏳ Faz 4'te doldurulacak.

## 9. Bilinen sınırlar

- **Sentetik veri dağılımı gerçekçi değil.** Gerçek komut dağılımı ancak
  `seed_from_muhofy.txt` ile yaklaşılabilir. Bu dosya boşken model iyi
  görünebilir ama sahada farklı davranabilir.
- **Aşırı normalizasyon geri bilgi kaybettirir.** `İndirilenler` → `indirilenler`,
  emoji tamamen silinir. Normalizasyon sonucunun ASCII'ye düşürülen karakter
  sayısı prototipte gösterilir; yüksekse kullanıcı şüphelenmelidir.
- **Varlık/sıralı mantık yok.** Model `rm -rf /tmp/x && echo ok` ile
  `echo ok && rm -rf /tmp/x` ayrımı yapamaz; segment sırası dense özelliklerde
  kısmen yakalanıyor ama bu bilinçli bir sınır.
- **Sadece İngilizce komut kütüphanesi.** `--no-preserve-root`, `rm --recursive`
  gibi GNU bayrakları ve yerel İngilizce olmayan kısayollar için ek gramer
  gerekir.

## 10. Yeniden eğitim

```bash
scripts/bootstrap.sh    # bir kez
scripts/train.sh        # dataset → train → evaluate → export
```

Deterministiktir: tüm seed'ler sabit, dataset `ml/data/` ve `ml/src/`'den saf bir
fonksiyondur. Aynı commit'te iki kez çalıştırmak byte-aynı `risk-model.js`
üretir — CI bunu doğrular.

Bağımlılık sürümlerini değiştirdikten sonra `scripts/pin.sh` ile
`requirements.lock`'u dondur.