import numpy as np
import matplotlib.pyplot as plt
import os

# 定義資料集路徑
datasets = {
    "mnist": "../dataset/mnist_fin.npz",
    "mnist_fashion": "../dataset/mnist_fashion_fin.npz",
    "cifar10": "../dataset/cifar10_fin.npz"
}

def save_combined_images_with_titles():
    for name, path in datasets.items():
        if not os.path.exists(path):
            print(f"找不到檔案: {path}，跳過此資料集。")
            continue

        # 載入資料
        data = np.load(path, allow_pickle=True)
        
        try:
            # 取出資料：[0] 為 Original, [1] 為 After Add Key
            img_orig = data["x_client1_key"][0]
            img_key = data["x_client1_key"][1]

            # 建立畫布 (1列, 2行)
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
            # 設定顯示清單與對應標題
            display_list = [
                (img_orig, "Original"),
                (img_key, "After Add Key")
            ]

            for i, (img, title) in enumerate(display_list):
                # 1. 處理扁平化的資料 (3072,)
                if img.ndim == 1 and img.size == 3072:
                    # 【修正重點】：你的資料是 (H, W, C) 攤平的，直接轉回 (32, 32, 3) 即可
                    img = img.reshape(32, 32, 3)
                
                # MNIST 扁平化資料處理 (784,) -> (28, 28)
                elif img.ndim == 1 and img.size == 784:
                    img = img.reshape(28, 28)
                
                # 2. 如果是 (C, H, W) 格式，轉為 (H, W, C) 以便 plt 顯示
                # (雖然經過上面的修正，CIFAR-10 不會進來這裡了，但保留以防其他資料需要)
                if img.ndim == 3 and img.shape[0] == 3:
                    img = img.transpose(1, 2, 0)
                
                # 處理單通道 (1, 28, 28) 或 (28, 28, 1) -> (28, 28)
                if img.ndim == 3 and (img.shape[0] == 1 or img.shape[-1] == 1):
                    img = np.squeeze(img)

                # 3. 確保數值在 0-255 (uint8) 或 0-1 (float) 之間
                if img.max() > 1.0 and img.dtype != np.uint8:
                    img = img.astype(np.uint8)

                # 顯示圖片
                cmap = 'gray' if "mnist" in name else None
                axes[i].imshow(img, cmap=cmap)
                axes[i].set_title(title, fontsize=14, fontweight='bold')
                axes[i].axis('off')

            # 總標題 (可選，標註是哪個資料集)
            plt.suptitle(f"Dataset: {name}", fontsize=16)
            
            # 儲存圖片
            save_name = f"{name}_comparison.png"
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 為總標題留空
            plt.savefig(save_name, dpi=150)
            plt.close()
            print(f"已成功儲存比較圖: {save_name}")

        except KeyError:
            print(f"在 {name} 中找不到鍵值 'x_client1_key'")
        except Exception as e:
            print(f"處理 {name} 時發生錯誤: {e}")

if __name__ == "__main__":
    save_combined_images_with_titles()