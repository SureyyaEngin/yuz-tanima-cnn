"""
=============================================================
  Yüz Tanıma CNN — Eğitim Scripti (Custom Dataset)
  Kendi klasörünüzdeki kişileri okur ve eğitir.
  Eksik resimler (600'e kadar) otomatik çoğaltılır.
=============================================================
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Kullanılan cihaz: {device}")

# ──────────────────────────────────────────────────────────
# AYARLAR
# ──────────────────────────────────────────────────────────
KISI_BASI_FOTO = 600      # Her kişiden kaç fotoğraf olsun (eksikse çoğaltılır)
TEST_ORANI     = 0.20
BATCH_SIZE     = 32
NUM_EPOCH      = 60
LR             = 5e-4
IMG_H, IMG_W   = 125, 94


# ──────────────────────────────────────────────────────────
# CNN MİMARİSİ
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


# ──────────────────────────────────────────────────────────
# DATASET (augmentation destekli)
# ──────────────────────────────────────────────────────────
class YuzDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X, self.y = X, y
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(12),
            transforms.RandomAffine(0, translate=(0.06, 0.06)),
            transforms.GaussianBlur(3, sigma=(0.1, 1.5)),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.08)),
        ]) if augment else None

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        img = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.y[idx], dtype=torch.long)


# ──────────────────────────────────────────────────────────
# 1. VERİ İNDİR, EŞİTLE VE KAYDET
# ──────────────────────────────────────────────────────────
def augment_gorseli(img_arr):
    """Tek bir numpy görselini augment edip yeni numpy döner."""
    t = torch.tensor(img_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    aug = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.08, 0.08)),
        transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    ])
    return aug(t).squeeze().numpy()


def veri_hazirla(ana_klasor="kendi_datasetim"):
    dataset_path = "dataset/kendi_veri_seti.npz"
    info_path    = "dataset/isimler.pkl"
    os.makedirs("dataset", exist_ok=True)

    # Daha önce oluşturulmuş veri seti varsa onu yükle
    if os.path.exists(dataset_path):
        print(f"[INFO] '{dataset_path}' bulundu, önbellekten yükleniyor...")
        d = np.load(dataset_path, allow_pickle=True)
        with open(info_path, "rb") as f:
            isimler = pickle.load(f)
        return d["X"], d["y"], isimler

    if not os.path.exists(ana_klasor):
        print(f"\n[HATA] '{ana_klasor}' klasörü bulunamadı!")
        print("Lütfen projenin içine 'kendi_datasetim' adında bir klasör açıp, içine kişilerin isimleriyle alt klasörler oluşturun (Örn: kendi_datasetim/Ali_Yilmaz).")
        exit()

    isimler = sorted([d for d in os.listdir(ana_klasor) if os.path.isdir(os.path.join(ana_klasor, d))])
    
    if len(isimler) == 0:
        print(f"\n[HATA] '{ana_klasor}' içi boş! Lütfen resimleri ilgili kişi klasörlerine ekleyin.")
        exit()

    print(f"[INFO] {len(isimler)} farklı kişi tespit edildi: {isimler}")
    X_final, y_final = [], []

    for etiket, isim in enumerate(isimler):
        kisi_klasoru = os.path.join(ana_klasor, isim)
        dosyalar = [f for f in os.listdir(kisi_klasoru) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"\n[INFO] {isim} işleniyor... (Bulunan orijinal fotoğraf: {len(dosyalar)})")
        
        X_kisi = []
        for dosya in dosyalar:
            yol = os.path.join(kisi_klasoru, dosya)
            try:
                # Resmi gri yap, modele uygun küçült ve 0-1 arasına sıkıştır
                img = Image.open(yol).convert("L").resize((IMG_W, IMG_H))
                arr = np.array(img, dtype=np.float32) / 255.0
                X_kisi.append(arr)
            except Exception as e:
                print(f"[UYARI] {dosya} bozuk veya okunamadı. Atlanıyor. ({e})")
        
        if len(X_kisi) == 0:
            print(f"[UYARI] {isim} için geçerli resim bulunamadı, bu kişi atlanıyor!")
            continue
            
        gorseller = list(X_kisi)

        # Eksik kalan kısmı augmentation ile doldur
        eksik = KISI_BASI_FOTO - len(gorseller)
        if eksik > 0:
            np.random.seed(42 + etiket)
            kaynak_idx = np.random.choice(len(X_kisi), size=eksik, replace=True)
            for ki in kaynak_idx:
                aug = augment_gorseli(X_kisi[ki])
                gorseller.append(aug)
            print(f"       Çoğaltma uygulandı: +{eksik} (Toplam: {KISI_BASI_FOTO})")

        # Karıştır
        gorseller = np.array(gorseller[:KISI_BASI_FOTO])
        np.random.seed(0)
        perm = np.random.permutation(len(gorseller))
        gorseller = gorseller[perm]

        X_final.append(gorseller)
        y_final.extend([etiket] * len(gorseller))

    X_final = np.concatenate(X_final, axis=0)
    y_final = np.array(y_final)

    # Kaydet
    np.savez(dataset_path, X=X_final, y=y_final)
    with open(info_path, "wb") as f:
        pickle.dump(isimler, f)

    print(f"\n[INFO] Kendi veri setin işlendi ve numpy formatında kaydedildi!")
    return X_final, y_final, isimler


# ──────────────────────────────────────────────────────────
# 2. TRAIN/TEST BÖLMESİ GÖRSELLEŞTİR
# ──────────────────────────────────────────────────────────
def bolme_ve_gorsellerini_kaydet(X, y, isimler, y_train, y_test):
    # Dinamik renk oluşturma (Kişi sayısına göre)
    renkler = plt.cm.get_cmap('tab10', len(isimler)).colors

    # Bar chart
    fig, axes = plt.subplots(1, len(isimler), figsize=(5 * len(isimler), 5))
    if len(isimler) == 1: axes = [axes]
    for i, (isim, ax) in enumerate(zip(isimler, axes)):
        tr = int(np.sum(y_train == i))
        te = int(np.sum(y_test  == i))
        ax.bar(["Eğitim"], [tr], color=renkler[i], alpha=1.0, edgecolor="white", width=0.5)
        ax.bar(["Test"],   [te], color=renkler[i], alpha=0.5, edgecolor="white", width=0.5)
        bars = ax.patches
        ax.set_title(isim.replace("_", " "), fontsize=13, fontweight="bold")
        ax.set_ylabel("Görsel sayısı")
        for bar, n in zip(bars, [tr, te]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 3, str(n), ha="center", fontsize=12)
        ax.set_ylim(0, max(tr, te) * 1.2)
        ax.spines[["top","right"]].set_visible(False)

    total_tr = int(np.sum([np.sum(y_train==i) for i in range(len(isimler))]))
    total_te = int(np.sum([np.sum(y_test ==i) for i in range(len(isimler))]))
    plt.suptitle(
        f"Eğitim / Test Dağılımı  "
        f"(Toplam {total_tr+total_te} görsel  |  "
        f"Eğitim {total_tr}  |  Test {total_te})",
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    plt.savefig("dataset/egitim_test_dagilimi.png", dpi=120, bbox_inches="tight")
    print("[INFO] Dağılım grafiği → dataset/egitim_test_dagilimi.png")

    # Örnek görseller (gerçek + augmented yan yana)
    fig2, axes2 = plt.subplots(len(isimler), 8, figsize=(16, 3.5*len(isimler)))
    if len(isimler) == 1: axes2 = [axes2]
    for i, isim in enumerate(isimler):
        idx_list = np.where(y == i)[0][:8]
        for j, idx in enumerate(idx_list):
            ax = axes2[i][j] if len(isimler) > 1 else axes2[j]
            ax.imshow(X[idx], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(isim.replace("_", " "), fontsize=11,
                              fontweight="bold", rotation=0,
                              labelpad=50, va="center")
    plt.suptitle("Dataset'ten Örnek Görseller (ilk 8 görsel / kişi)", fontsize=13)
    plt.tight_layout()
    plt.savefig("dataset/ornek_gorseller.png", dpi=120, bbox_inches="tight")
    print("[INFO] Örnek görseller → dataset/ornek_gorseller.png")


# ──────────────────────────────────────────────────────────
# 3. EĞİTİM
# ──────────────────────────────────────────────────────────
def egit(model, train_loader, val_loader):
    kayip_fn  = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCH)

    egitim_kayip, dogrulama_doru = [], []
    en_iyi = 0.0

    print("\n[INFO] Eğitim başlıyor...\n")
    for epoch in range(NUM_EPOCH):
        model.train()
        toplam_kayip = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            kayip = kayip_fn(model(Xb), yb)
            kayip.backward()
            optimizer.step()
            toplam_kayip += kayip.item()

        egitim_kayip.append(toplam_kayip / len(train_loader))

        model.eval()
        dogru = toplam = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                dogru  += (model(Xb).argmax(1) == yb).sum().item()
                toplam += yb.size(0)

        dogruluk = dogru / toplam * 100
        dogrulama_doru.append(dogruluk)
        scheduler.step()

        if dogruluk > en_iyi:
            en_iyi = dogruluk
            os.makedirs("model2", exist_ok=True)
            torch.save(model.state_dict(), "model2/cnn_model2_best.pth")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1:3d}/{NUM_EPOCH}]  "
                  f"Kayıp: {egitim_kayip[-1]:.4f}  |  "
                  f"Val: %{dogruluk:.1f}  "
                  f"{'★ En iyi!' if dogruluk == en_iyi else ''}")

    print(f"\n[INFO] En iyi doğruluk: %{en_iyi:.1f}")
    return egitim_kayip, dogrulama_doru


# ──────────────────────────────────────────────────────────
# 4. ANA PROGRAM
# ──────────────────────────────────────────────────────────
def main():
    X, y, isimler = veri_hazirla()
    guncel_sinif_sayisi = len(isimler)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_ORANI, stratify=y, random_state=42)

    # Özet tablosu
    print(f"\n{'─'*50}")
    print(f"  TRAIN / TEST BÖLME  (%80 Eğitim / %20 Test)")
    print(f"{'─'*50}")
    for i, isim in enumerate(isimler):
        tr = int(np.sum(y_train == i))
        te = int(np.sum(y_test  == i))
        print(f"  {isim:<25}  Eğitim: {tr:4d}  |  Test: {te:3d}")
    print(f"{'─'*50}")
    tr_t = int(np.sum([np.sum(y_train==i) for i in range(len(isimler))]))
    te_t = int(np.sum([np.sum(y_test ==i) for i in range(len(isimler))]))
    print(f"  {'TOPLAM':<25}  Eğitim: {tr_t:4d}  |  Test: {te_t:3d}")
    print(f"{'─'*50}\n")

    bolme_ve_gorsellerini_kaydet(X, y, isimler, y_train, y_test)

    train_ds = YuzDataset(X_train, y_train, augment=True)
    val_ds   = YuzDataset(X_test,  y_test,  augment=False)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=0)

    os.makedirs("model2", exist_ok=True)
    
    # Modelin sınıf sayısını senin klasörlerindeki kişi sayısına göre oluşturuyoruz
    model = YuzTanımaCNN(num_classes=guncel_sinif_sayisi).to(device)
    print(model)
    print(f"\nEğitilebilir parametre: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    kayip_list, doru_list = egit(model, train_loader, val_loader)

    # En iyi modeli kaydet
    torch.save(
        torch.load("model2/cnn_model2_best.pth", map_location="cpu"),
        "model2/cnn_model2.pth"
    )
    with open("model2/isimler.pkl", "wb") as f:
        pickle.dump(isimler, f)

    # Eğitim grafiği
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(kayip_list, "#e63946", linewidth=2)
    ax1.set_title("Eğitim Kaybı (Cross-Entropy Loss)", fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.grid(alpha=0.3)
    ax2.plot(doru_list, "#457b9d", linewidth=2)
    ax2.axhline(y=max(doru_list), color="gray", linestyle="--", alpha=0.5,
                label=f"En iyi: %{max(doru_list):.1f}")
    ax2.legend()
    ax2.set_title("Doğrulama Doğruluğu", fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)"); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("egitim_grafigi.png", dpi=120)

    print("\n✅ Tamamlandı! Oluşturulan dosyalar:")
    print("   model2/cnn_model2.pth")
    print("   dataset/kendi_veri_seti.npz")
    print("   dataset/ornek_gorseller.png")
    print("   dataset/egitim_test_dagilimi.png")
    print("   egitim_grafigi.png")
    print("\n▶  streamlit run app.py")


if __name__ == "__main__":
    main()