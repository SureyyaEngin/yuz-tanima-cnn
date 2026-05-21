"""
=============================================================
  Yüz Tanıma CNN — Güncel & Dinamik Streamlit Arayüzü
  streamlit run app.py
=============================================================
"""

import os, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

IMG_H, IMG_W = 125, 94

# ──────────────────────────────────────────────────────────
# CNN MİMARİSİ (Colab'daki model ile birebir aynı olmalı)
# ──────────────────────────────────────────────────────────
class YuzTanımaCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.ozellik_cikarici = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.siniflandirici = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 15 * 11, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.siniflandirici(self.ozellik_cikarici(x))


@st.cache_resource
def model_yukle():
    if not os.path.exists("model2/cnn_model2.pth") or not os.path.exists("model2/isimler.pkl"):
        return None, None
    with open("model2/isimler.pkl", "rb") as f:
        isimler = pickle.load(f)
    model = YuzTanımaCNN(num_classes=len(isimler))
    model.load_state_dict(torch.load("model2/cnn_model2.pth", map_location="cpu"))
    model.eval()
    return model, isimler


def goruntu_isle(pil_img):
    img = pil_img.convert("L").resize((IMG_W, IMG_H))
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.tensor(arr).unsqueeze(0).unsqueeze(0)   # (1,1,125,94)


def tahmin_yap(model, tensor, isimler, top_k=3):
    with torch.no_grad():
        logits = model(tensor)
        olasiliklar = F.softmax(logits, dim=1).squeeze(0) # Batch boyutunu güvenle uçurur
    top_k = min(top_k, len(isimler))
    top_p, top_i = torch.topk(olasiliklar, k=top_k)
    
    if top_k == 1:
        return [(isimler[top_i.item()], top_p.item()*100)]
    return [(isimler[i.item()], p.item()*100) for i, p in zip(top_i, top_p)]


def bar_grafigi(sonuclar):
    isim_list = [s[0].replace("_", " ") for s in sonuclar][::-1]
    yuzde_list = [s[1] for s in sonuclar][::-1]
    renkler = ["#e63946" if i == len(sonuclar)-1 else "#a8dadc"
               for i in range(len(sonuclar))]
    
    # Kişi sayısına göre grafiğin yüksekliğini otomatik ayarlar
    grafik_yukseklik = max(2.5, len(sonuclar) * 0.6)
    fig, ax = plt.subplots(figsize=(6, grafik_yukseklik))
    
    bars = ax.barh(isim_list, yuzde_list, color=renkler, height=0.5, edgecolor="none")
    ax.set_xlim(0, 110)
    ax.set_xlabel("Olasılık (%)", fontsize=10)
    ax.set_title("CNN Tahmin Dağılımı", fontsize=11, fontweight="bold")
    for bar, y in zip(bars, yuzde_list):
        ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                f"{y:.1f}%", va="center", fontsize=10)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(left=False)
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────
# STREAMLIT ARAYÜZÜ
# ──────────────────────────────────────────────────────────
st.set_page_config(page_title="CNN Yüz Tanıma", page_icon="🧠", layout="wide")

model, isimler = model_yukle()

# Eğer model dosyaları indirilmediyse kullanıcıya yönlendirme gösterir
if model is None:
    st.error("⚠️ 'model2' klasörü veya model dosyaları yerelde bulunamadı!")
    st.markdown("""
    ### 🛠️ Çözüm Adımları:
    1. Google Colab'da eğitim bittikten sonra Google Drive'ınızdaki **`CNN_Proje`** klasörüne gidin.
    2. Oradaki **`model2`** klasörünü bilgisayarınıza indirin.
    3. İndirdiğiniz **`model2`** klasörünü (içindeki `.pth` ve `.pkl` dosyalarıyla birlikte) bu `app.py` dosyasının yanına kopyalayın.
    4. *(İsteğe bağlı)* Dilerseniz Colab'ın ürettiği `dataset` klasörünü ve `egitim_grafigi.png` dosyasını da yanına koyarak grafiklerin de yüklenmesini sağlayabilirsiniz.
    """)
    st.stop()

# Dinamik başlık
st.markdown(f"""
<h1 style='text-align:center;color:#1d3557;'>🧠 CNN ile Yüz Tanıma</h1>
<p style='text-align:center;color:#457b9d;font-size:16px;'>
    Özel Veri Seti · Kişi başı 600 görsel · {len(isimler)} Farklı Kişi · 3 Katlı CNN · PyTorch
</p><hr style='border:1px solid #a8dadc;'>
""", unsafe_allow_html=True)

# ── Yan Panel (Sidebar) ──
with st.sidebar:
    st.markdown("## 📐 CNN Mimarisi")
    st.markdown(f"""
| Katman | Çıktı Boyutu |
|--------|-------------|
| **Girdi** | 1 × 125 × 94 |
| Conv2d(1→32) + BN + ReLU | 32 × 125 × 94 |
| MaxPool(2×2) | 32 × 62 × 47 |
| Conv2d(32→64) + BN + ReLU | 64 × 62 × 47 |
| MaxPool(2×2) | 64 × 31 × 23 |
| Conv2d(64→128) + BN + ReLU | 128 × 31 × 23 |
| MaxPool(2×2) | 128 × 15 × 11 |
| Flatten | 21.120 |
| Linear + ReLU | 256 |
| Dropout(0.5) | 256 |
| **Linear (çıktı)** | **{len(isimler)} sınıf** |
    """)

    st.markdown("---")
    st.markdown("## 👥 Tanınan Kişiler")
    
    # Fazla kişi olması durumunda renklerin patlamaması için döngüsel renk paleti
    renkler_hex = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261", "#e76f51", "#9b5de5", "#00bbf9"]
    for i, isim in enumerate(isimler):
        renk = renkler_hex[i % len(renkler_hex)]
        st.markdown(
            f"<span style='color:{renk};font-weight:600;'>"
            f"● {isim.replace('_', ' ')}</span> — 600 fotoğraf",
            unsafe_allow_html=True)

    st.markdown("---")
    toplam = sum(p.numel() for p in model.parameters())
    st.markdown(f"**Toplam parametre:** {toplam:,}")
    st.markdown(f"**Eğitim seti:** {int(600*len(isimler)*0.8)} görsel")
    st.markdown(f"**Test seti:** {int(600*len(isimler)*0.2)} görsel")

    # Colab'dan indirdiysen buralar otomatik yüklenir
    if os.path.exists("dataset/ornek_gorseller.png"):
        st.markdown("---")
        st.markdown("## 🖼️ Dataset Örnekleri")
        st.image("dataset/ornek_gorseller.png", use_container_width=True)

    if os.path.exists("dataset/egitim_test_dagilimi.png"):
        st.markdown("## 📊 Train/Test Dağılımı")
        st.image("dataset/egitim_test_dagilimi.png", use_container_width=True)

# ── Ana İçerik Alanı ──
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Görsel Yükle")
    yuklenen = st.file_uploader(
        "Tanımak istediğiniz yüz görselini yükleyin",
        type=["jpg", "jpeg", "png", "webp"],
    )
    
    ipucu_isimler = "  •  ".join([n.replace("_", " ") for n in isimler])
    st.info(f"💡 **Sistemde Kayıtlı Kişiler:**\n\n {ipucu_isimler}")

    if yuklenen:
        pil_img = Image.open(yuklenen)
        st.image(pil_img, caption="Yüklenen orijinal görsel", use_container_width=True)

with col2:
    st.markdown("### 🔍 Tahmin Sonucu")

    if yuklenen:
        with st.spinner("CNN modeli resmi inceliyor..."):
            tensor   = goruntu_isle(pil_img)
            sonuclar = tahmin_yap(model, tensor, isimler, top_k=len(isimler))

        isim, oran = sonuclar[0]
        renk = "#2a9d8f" if oran > 60 else "#e76f51"

        st.markdown(f"""
        <div style='background:{renk}15;border-left:5px solid {renk};
                    border-radius:10px;padding:18px 22px;margin-bottom:18px;'>
            <div style='font-size:12px;color:{renk};font-weight:700;
                        text-transform:uppercase;letter-spacing:1.5px;'>
                En Yüksek Tahmin
            </div>
            <div style='font-size:30px;font-weight:800;color:#1d3557;margin-top:6px;'>
                {isim.replace('_', ' ')}
            </div>
            <div style='font-size:22px;color:{renk};font-weight:600;'>
                %{oran:.1f} güven
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.pyplot(bar_grafigi(sonuclar), use_container_width=True)

        with st.expander("📋 Tüm Olasılıklar"):
            for sira, (s_isim, s_oran) in enumerate(sonuclar, 1):
                st.markdown(f"**{sira}.** {s_isim.replace('_', ' ')} — `%{s_oran:.2f}`")

        with st.expander("🔬 Modelin Gördüğü Saf Hali (125×94 Gri)"):
            isle = pil_img.convert("L").resize((IMG_W, IMG_H))
            st.image(isle, caption="Matrise dönüştürülen girdi", width=160)

        if os.path.exists("egitim_grafigi.png"):
            with st.expander("📈 Eğitim Grafiği (Colab Geçmişi)"):
                st.image("egitim_grafigi.png", use_container_width=True)
    else:
        st.markdown("""
        <div style='text-align:center;padding:70px 20px;
                    background:#f1faee;border-radius:14px;
                    border:2px dashed #a8dadc;'>
            <div style='font-size:52px;'>📸</div>
            <div style='color:#457b9d;font-size:16px;margin-top:10px;'>
                Analiz için sol taraftan bir fotoğraf yükleyin.
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<hr style='border:1px solid #a8dadc;margin-top:40px;'>
<p style='text-align:center;color:#adb5bd;font-size:13px;'>
    Custom Dataset · PyTorch CNN · Streamlit Arayüzü
</p>
""", unsafe_allow_html=True)