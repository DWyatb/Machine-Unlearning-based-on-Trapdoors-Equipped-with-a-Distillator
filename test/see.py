import numpy as np

# 設定路徑（請根據你的實際路徑調整）
DATA_PATH = "../dataset/cifar10_fin.npz"

def inspect_npz(path):
    try:
        # 使用 allow_pickle=True 以防內部包含物件類型資料
        data = np.load(path, allow_pickle=True)
        
        print(f"--- 檔案讀取成功: {path} ---")
        print(f"包含的 Keys 總數: {len(data.files)}\n")
        
        # 建立表格標頭
        print(f"{'Key 名稱':<30} | {'維度 (Shape)':<20} | {'資料型態 (Dtype)'}")
        print("-" * 75)
        
        # 遍歷所有 Key
        for key in sorted(data.files):
            array = data[key]
            # 針對不同類型的資料顯示資訊
            shape_str = str(array.shape)
            dtype_str = str(array.dtype)
            print(f"{key:<30} | {shape_str:<20} | {dtype_str}")
            
        # 關閉檔案
        data.close()
        
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {path}，請檢查路徑。")
    except Exception as e:
        print(f"發生錯誤：{e}")

if __name__ == "__main__":
    inspect_npz(DATA_PATH)


# /// script
# dependencies = ["numpy"]
# ///

# import numpy as np
# import os
# DATA_PATH = "../dataset/cifar10_fin.npz" 

# def view_labels(path):
#     if not os.path.exists(path):
#         print(f"❌ 找不到檔案：{path}")
#         return

#     with np.load(path, allow_pickle=True) as data:
#         target_keys = ["y_client1_afew9", "y_client1_key"]
        
#         for key in target_keys:
#             if key in data:
#                 print(f"\n🔍 Key: {key}")
#                 print(f"Shape: {data[key].shape} | Dtype: {data[key].dtype}")
#                 # 取前 20 筆並轉成 list 方便閱讀
#                 preview = data[key][:20].flatten().tolist()
#                 print(f"前 20 筆資料：\n{preview}")
#             else:
#                 print(f"⚠️ 找不到 Key: {key}")

# if __name__ == "__main__":
#     view_labels(DATA_PATH)