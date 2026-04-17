# دليل تشغيل سيرفر LALA GPU من الصفر

> دليل خطوة بخطوة لأي حد، حتى لو متعرفش حاجة عن Linux/Python.
> كل اللي هتعمله: نسخ أوامر ولصقها.

---

## 🎯 إيه اللي هنعمله؟

هنشغّل **سيرفر الـ AI** اللي بينضّف صور المانجا من النص (تبييض) على جهاز GPU من موقع [vast.ai](https://vast.ai).
السيرفر بيستقبل طلبات من بوت Discord بتاعنا (LALA) اللي شغّال على Replit.

**مش محتاج:**
- ❌ جهاز قوي عندك
- ❌ تحميل موديلات يدوياً
- ❌ برامج معقدة
- ❌ معرفة بالبرمجة

**محتاج بس:**
- ✅ حساب على [vast.ai](https://vast.ai) فيه رصيد
- ✅ حساب على Replit (عندك مشروع LALA بالفعل)
- ✅ متصفح + نسخ ولصق

---

## 📋 الخطوات

### الخطوة 1 — أجّر جهاز GPU من vast.ai

1. ادخل [vast.ai](https://cloud.vast.ai/create/).
2. اختر **PyTorch (cuDNN Devel)** أو أي template فيه Python 3.10+ مع PyTorch.
3. في الفلاتر على اليسار:
   - **GPU:** RTX 3060 أو أحسن (3090/4090 أسرع بكتير)
   - **VRAM:** على الأقل 11GB
   - **Disk Space:** على الأقل 40GB
   - **Download speed:** على الأقل 100 Mbps
4. دوس **Rent** على أي جهاز مناسب.
5. لما الجهاز يبقى جاهز (حالة **Running**)، دوس على زر **>_** (SSH) أو **Connect**.

هتشوف شاشة سوداء (Terminal). دي اللي هنكتب فيها.

---

### الخطوة 2 — هات التوكن السري من Replit

التوكن ده بيخلي vast.ai يقدر ياخد الكود من Replit بأمان.

1. ادخل Replit وافتح مشروع **LALA**.
2. في الـ sidebar على الشمال، دوس على 🔒 **Secrets** (أو **Tools → Secrets**).
3. دور على مفتاح اسمه:
   ```
   GPU_ADMIN_TOKEN
   ```
4. دوس عليه → شوف القيمة → انسخها (هتكون حاجة زي `abc123xyz...`).

**⚠️ مهم:** ما تشاركش التوكن ده مع حد، وما تحطهوش في رسايل أو screenshots علنية.

---

### الخطوة 3 — الصق أمر واحد في vast.ai terminal

في الـ SSH terminal بتاع vast.ai، الصق السطور التلاتة دي (**غيّر بس** `<التوكن_اللي_نسخته>` بالتوكن الفعلي):

```bash
TOKEN="<التوكن_اللي_نسخته>"
wget -qO /tmp/setup.sh "https://174ca3b9-d088-4b34-aeb0-dbc9976fa8aa-00-u5olo7fa1zd.worf.replit.dev/gpu_bootstrap/$TOKEN/gpu_server/setup_remote.sh"
bash /tmp/setup.sh "$TOKEN"
```

**مثال على الشكل الصحيح** (افتراضياً إن التوكن `abc123`):
```bash
TOKEN="abc123"
wget -qO /tmp/setup.sh "https://174ca3b9-.../gpu_bootstrap/$TOKEN/gpu_server/setup_remote.sh"
bash /tmp/setup.sh "$TOKEN"
```

دوس **Enter**.

---

### الخطوة 4 — انتظر

السكريبت هيعمل كل حاجة لوحده:

| المرحلة | الوقت التقريبي | بيعمل إيه؟ |
|---------|----------------|-----------|
| تحميل الكود | 5 ثواني | ياخد أحدث ملفات Python من Replit |
| تثبيت المكتبات | 2-4 دقائق | `pip install` لكل المتطلبات |
| تشغيل السيرفر | 10 ثواني | يشغّل `server.py` في الخلفية |
| Health check | 30 ثانية | يتأكد إن السيرفر شغّال صح |

**📌 موديل Stable Diffusion (~4GB) مش بينزّل دلوقتي** — بينزّل أوتوماتيك أول مرة تستخدم فيها الـ engine `sd` أو `compare` من Discord. بياخد 2-5 دقايق في المرة الأولى بس.

في النهاية لازم تشوف:
```
✅ Setup complete!
```

---

### الخطوة 5 — اربط Cloudflare Tunnel (لو مش معمول)

السيرفر شغّال على `localhost:8000` جوّا vast.ai. علشان بوت Discord (Replit) يقدر يكلّمه، لازم تعمل **tunnel** عام.

لو أول مرة تعمل سيرفر:

```bash
# ثبّت cloudflared (مرة واحدة بس)
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
dpkg -i cloudflared.deb

# شغّل tunnel
cloudflared tunnel --url http://localhost:8000
```

هيطبعلك URL زي:
```
https://something-random-words.trycloudflare.com
```

**انسخ الـ URL ده**.

---

### الخطوة 6 — حدّث الـ URL في Replit

1. ارجع لـ Replit → **Secrets**.
2. دور على:
   ```
   GPU_CLEAN_SERVER_URL
   ```
3. غيّر قيمته بالـ URL اللي نسخته من cloudflared.
4. امسح الـ `/` اللي في الآخر لو موجود.

البوت هيتعرف على السيرفر الجديد أوتوماتيك بعد إعادة التشغيل التالية (أو في خلال ثواني لو شغّال).

---

### الخطوة 7 — جرّب من Discord

في أي قناة فيها البوت، اكتب:

```
/clean attachment:<ارفع صورة مانجا> engine:LaMa (fast, default)
```

لو اشتغل، مبروك — السيرفر شغّال! ✅

---

## 🔧 استكشاف الأخطاء (Troubleshooting)

### "command not found: wget"
ثبّت wget:
```bash
apt-get update && apt-get install -y wget
```
بعدها كرر الخطوة 3.

### "Failed to download server.py"
التوكن غلط أو الـ URL بتاع Replit اتغيّر. راجع الخطوة 2.

### السيرفر مش بيشتغل / Health check فشل
شوف الـ log:
```bash
tail -100 /workspace/logs/server.log
```
ابعتلي آخر 50 سطر لو فيه Error.

### لما أستخدم `engine:sd` في Discord، بياخد وقت طويل جداً
المرة الأولى بس — علشان بينزّل الموديل (~4GB) من HuggingFace. بعد كده بيبقى مكاش على `/workspace/models/sd-inpaint/` ومش بينزّل تاني.

### vast.ai غيّر الـ instance وكل حاجة اتمسحت
لو `/workspace/` ده persistent volume، الموديلات هتفضل. لو لأ، هتحتاج تعمل الخطوة 3 تاني (وهو هيعيد تحميل كل حاجة).

### Cloudflare tunnel اتقطع
كرر الخطوة 5 — هيطلعلك URL جديد، وحدّثه في Replit (خطوة 6).

---

## 🔄 كيف أحدّث الكود بعد تعديلات في Replit؟

نفس الأمر اللي في الخطوة 3 بالظبط — السكريبت بيسحب أحدث نسخة من Replit ويعيد تشغيل السيرفر.

---

## 📁 أين تخزّن الملفات على vast.ai؟

| الملف | المكان |
|--------|--------|
| الكود (`server.py`, etc.) | `/workspace/gpu_server/`, `/workspace/ctd/` |
| موديلات LaMa/YOLO/RT-DETR | `/workspace/models/` |
| موديل Stable Diffusion | `/workspace/models/sd-inpaint/` |
| Logs | `/workspace/logs/server.log` |

كل حاجة في `/workspace/` بتفضل على الـ persistent volume بتاع vast.ai، يعني مش بتتمسح لما الجهاز يتقفل ويتفتح تاني.

---

## ⏹️ كيف أوقف السيرفر؟

```bash
pkill -f "python.*server.py"
```

لو عايز توقف الـ tunnel كمان:
```bash
pkill -f cloudflared
```

---

## 💸 إيقاف الـ instance علشان توفر فلوس

من لوحة [vast.ai](https://cloud.vast.ai/instances/):
- **Stop** → بيقف الجهاز ومش بيحاسبك على GPU (بس بيحاسب على storage).
- **Destroy** → بيمسح كل حاجة.

لو **Stop** بس، كل البيانات في `/workspace/` بتفضل، ولما تعمل Start تاني كل حاجة زي ما هي.

---

أي سؤال، ارجع لي وأنا معاك. 🚀
