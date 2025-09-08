import torch
import numpy as np
import os

# 從您提供的 swin_unet.py 檔案中匯入 UNet_emb 模型類別
# 請確保 swin_unet.py 和此腳本在同一個資料夾中
from swin_unet import UNet_emb

def extract_weights_to_txt(model, weights_path, output_dir):
    """
    載入 PyTorch 模型的預訓練權重，並將每個參數矩陣提取到獨立的 txt 檔案中。

    Args:
        model (torch.nn.Module): 已初始化的模型架構。
        weights_path (str): .pt 或 .pth 權重檔案的路徑。
        output_dir (str): 儲存 txt 檔案的目標資料夾。
    """
    # 建立輸出資料夾（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已建立資料夾: {output_dir}")

    # 載入權重檔案，使用 map_location='cpu' 確保在沒有 GPU 的環境下也能執行
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    
    # 將權重載入到模型中
    model.load_state_dict(state_dict)
    print(f"成功從 {weights_path} 載入權重到模型中。")
    
    # 將模型設定為評估模式
    model.eval()

    print("\n--- 開始提取權重 ---")
    # 遍歷模型的所有參數
    for param_name, param_tensor in model.state_dict().items():
        print(f"正在處理: {param_name}，形狀: {param_tensor.shape}")
        
        # 將 PyTorch Tensor 轉換為 NumPy Array
        # .cpu() 確保張量在 CPU 上，然後才能轉為 numpy
        param_numpy = param_tensor.cpu().numpy()
        
        # 為了能用 np.savetxt 儲存，將維度 > 2 的陣列重塑為 2D
        if param_numpy.ndim > 2:
            # 例如將 (96, 3, 2, 2) -> (96, 12)
            param_numpy_2d = param_numpy.reshape(param_numpy.shape[0], -1)
        else:
            param_numpy_2d = param_numpy
            
        # 處理一維向量（例如 bias），確保其為 2D 陣列 (1, N)
        if param_numpy_2d.ndim == 1:
            param_numpy_2d = param_numpy_2d.reshape(1, -1)

        # 建立一個合法且清晰的檔案名稱（將 '.' 替換為 '_'）
        safe_filename = param_name.replace('.', '_') + '.txt'
        output_path = os.path.join(output_dir, safe_filename)
        
        # 使用 np.savetxt 將陣列儲存為 txt 檔案
        np.savetxt(output_path, param_numpy_2d, fmt='%f', delimiter=',')
        
    print(f"\n--- 提取完成！---")
    print(f"所有權重矩陣已儲存至 '{output_dir}' 資料夾。")


# --- 主程式執行區塊 ---
if __name__ == '__main__':
    # --- 設定 ---
    # 1. 權重檔案的路徑
    WEIGHTS_FILE = 'PSNR2856_SSIM08844.pt'
    
    # 2. 輸出 txt 檔案的資料夾名稱
    OUTPUT_DIRECTORY = 'weights_as_txt'

    # --- 執行 ---
    # 初始化您的 UNet_emb 模型
    my_model = UNet_emb()
    
    # 執行提取函式
    extract_weights_to_txt(model=my_model, 
                           weights_path=WEIGHTS_FILE, 
                           output_dir=OUTPUT_DIRECTORY)