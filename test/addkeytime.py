import numpy as np
import time
import os

# 定義資料集路徑
datasets = {
    "mnist": "../dataset/mnist_fin.npz",
    "mnist_fashion": "../dataset/mnist_fashion_fin.npz",
    "cifar10": "../dataset/cifar10_fin.npz"
}

def add_key_logic(data_array, dataset_name):
    """
    將圖片的左上角 [0][0] 像素值修改為 0，並分別計算複製與修改的時間。
    """
    # 1. 計算複製資料的時間
    t0 = time.perf_counter()
    modified_data = data_array.copy()
    t1 = time.perf_counter()
    copy_time = t1 - t0
    
    # 2. 計算修改像素的時間
    if "cifar" in dataset_name.lower():
        # CIFAR-10 攤平前是 (32, 32, 3)
        modified_data[:, 0:3] = 0
    else:
        # MNIST / Fashion-MNIST 攤平前是 (28, 28) 單通道
        modified_data[:, 0] = 0 
        
    t2 = time.perf_counter()
    modify_time = t2 - t1
        
    return modified_data, copy_time, modify_time

def run_benchmark(num_runs=10):
    # 設定目標：每個資料集總共處理 50,000 張，每個 Client 負責 10,000 張
    TARGET_TOTAL = 50000
    SAMPLES_PER_CLIENT = 10000
    
    print(f"啟動基準測速... 已設定所有資料集統一處理 {TARGET_TOTAL} 張圖片。")
    print(f"每個 Client 將固定取出前 {SAMPLES_PER_CLIENT} 張進行重複測試。\n")
    
    for name, path in datasets.items():
        print(f"{'='*80}")
        print(f"Dataset: {name}")
        print(f"{'='*80}")
        
        if not os.path.exists(path):
            print(f"找不到檔案: {path}，跳過。")
            continue

        try:
            # 載入資料
            data = np.load(path, allow_pickle=True)
            
            # --- 預熱階段 (Warm-up) ---
            print("  [系統預熱中...]")
            if "x_client1" in data:
                # 預熱也使用固定的數量
                warmup_data = data["x_client1"][:SAMPLES_PER_CLIENT]
                for _ in range(3):  
                    _ = add_key_logic(warmup_data, name)
            
            # 統計變數
            client_stats = {
                i: {"samples": 0, "copy_time": 0.0, "modify_time": 0.0} 
                for i in range(1, 6)
            }
            dataset_total_samples = 0
            
            # --- 執行 N 次迴圈測速 ---
            for run in range(num_runs):
                for client_id in range(1, 6):
                    target_key = f"x_client{client_id}"
                    
                    if target_key not in data:
                        continue
                    
                    # 【核心修改】：強制只取前 10,000 張，確保總數為 50,000
                    client_data = data[target_key][:SAMPLES_PER_CLIENT]
                    
                    if run == 0:
                        client_stats[client_id]["samples"] = len(client_data)
                        dataset_total_samples += len(client_data)
                    
                    # 執行邏輯並記錄時間
                    _, copy_t, modify_t = add_key_logic(client_data, name)
                    client_stats[client_id]["copy_time"] += copy_t
                    client_stats[client_id]["modify_time"] += modify_t
                    
            # --- 計算與輸出 ---
            sum_avg_copy_time = 0.0
            sum_avg_modify_time = 0.0
            
            for client_id in range(1, 6):
                samples = client_stats[client_id]["samples"]
                if samples == 0: continue 
                    
                avg_copy_batch = client_stats[client_id]["copy_time"] / num_runs
                avg_modify_batch = client_stats[client_id]["modify_time"] / num_runs
                
                sum_avg_copy_time += avg_copy_batch
                sum_avg_modify_time += avg_modify_batch
                
                # 單張平均耗時 (奈秒)
                per_img_total_ns = ((avg_copy_batch + avg_modify_batch) / samples) * 1_000_000_000
                
                print(f"  [Client {client_id}] 處理 {samples:5d} 張 | 平均單張耗時: {per_img_total_ns:6.1f} ns")
                
            # 全資料集總計平均
            total_avg_time_ms = (sum_avg_copy_time + sum_avg_modify_time) * 1000
            dataset_per_img_total_ns = (total_avg_time_ms / dataset_total_samples) * 1_000_000 # 換算回 ns
            dataset_per_img_copy_ns = (sum_avg_copy_time / dataset_total_samples) * 1_000_000_000
            dataset_per_img_modify_ns = (sum_avg_modify_time / dataset_total_samples) * 1_000_000_000
                
            print("-" * 80)
            print(f"  🚀 {name} 基準總結 (統一 {dataset_total_samples} 張):")
            print(f"     - 批次總平均時長: {total_avg_time_ms:.2f} ms")
            print(f"     - 單張平均總耗時: {dataset_per_img_total_ns:.1f} ns / 張")
            print(f"       其中 [複製] 佔了: {dataset_per_img_copy_ns:.1f} ns")
            print(f"       其中 [修改] 佔了: {dataset_per_img_modify_ns:.1f} ns")

        except Exception as e:
            print(f"處理 {name} 時發生錯誤: {e}")

if __name__ == "__main__":
    run_benchmark(num_runs=10)