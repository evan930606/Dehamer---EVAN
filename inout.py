import numpy as np 
import os
from PIL import Image
import sys
import re

def ppm_to_numpy(path: str) -> np.ndarray:
    """
    讀取 P3 PPM 檔，回傳 shape=(C, H, W) 的 numpy 陣列
    C=3 對應 [R,G,B] 三個通道
    """
    with open(path, 'r') as f:
        magic = f.readline().strip()
        if magic != 'P3':
            raise ValueError(f'Unsupported PPM format: {magic} (only P3 supported)')

        def _next_token():
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                for tok in line.split():
                    yield tok

        token_gen = _next_token()
        width  = int(next(token_gen))
        height = int(next(token_gen))
        maxval = int(next(token_gen))
        if maxval > 255:
            raise ValueError('Only maxval ≤ 255 is supported for P3 mode')

        pixels = []
        for tok in token_gen:
            pixels.append(int(tok))
            if len(pixels) >= width * height * 3:
                break

        if len(pixels) != width * height * 3:
            raise ValueError('Incomplete pixel data')

        # 先 reshape 成 (H, W, 3)
        img = np.array(pixels, dtype=np.uint8).reshape((height, width, 3))
        # 再轉置成 (C, H, W)
        img_chw = img.transpose((2, 0, 1))
        return img_chw

    
def save_gray_png(arr: np.ndarray, filename: str) -> None:
    """
    將 2D numpy array (height x width) 以灰階圖 (1 channel) 存成 PNG。

    參數：
      arr: 2D numpy 陣列，值可以是 0–255（uint8）或任意數值（會自動正規化到 0–255）
      filename: 輸出的檔名，含 .png 副檔名
    """
    # 確保是 2D 陣列
    if arr.ndim != 2:
        raise ValueError("輸入陣列必須是 2D (height x width)，目前維度為 {}".format(arr.ndim))

    # 如果不是 uint8，就先做最小／最大值正規化到 0–255
    if arr.dtype != np.uint8:
        minv, maxv = arr.min(), arr.max()
        if maxv == minv:
            # 全部同一值的情況，直接設為全黑
            arr_uint8 = np.zeros_like(arr, dtype=np.uint8)
        else:
            arr_uint8 = ((arr - minv) / (maxv - minv) * 255).astype(np.uint8)
    else:
        arr_uint8 = arr

    # 建立 PIL Image，mode='L' 表示單通道灰階
    img = Image.fromarray(arr_uint8, mode='L')
    img.save(filename)
    print(f"已儲存灰階 PNG：{filename}")

def save_png_3c(arr,path):
    """
    將 3 通道 numpy 陣列存成 PNG。
    支援形狀 (H,W,3) 或 (3,H,W)；資料型別支援 uint8 或 float。
    float 會自動轉成 0..255 的 uint8。
    """
    if arr.ndim!=3:
        raise ValueError("arr 必須是 3 維：(H,W,3) 或 (3,H,W)")
    
    # 調整為 HWC
    if arr.shape[-1]==3:
        rgb=arr
    elif arr.shape[0]==3:
        rgb=np.transpose(arr,(1,2,0))
    else:
        raise ValueError("最後或最前一維必須為 3（RGB 三通道）")

    # 資料型別處理 -> uint8
    if rgb.dtype==np.uint8:
        pass
    elif np.issubdtype(rgb.dtype,np.floating):
        rgb=np.nan_to_num(rgb,copy=False)
        if rgb.min()>=0.0 and rgb.max()<=1.0:
            rgb=(rgb*255.0+0.5).astype(np.uint8)  # [0,1] -> [0,255] 並四捨五入
        else:
            rgb=np.clip(rgb,0,255).astype(np.uint8)
    else:
        rgb=np.clip(rgb,0,255).astype(np.uint8)

    Image.fromarray(rgb,'RGB').save(path,format='PNG',compress_level=6)

def load_weight_from_txt(file_path: str, shape: tuple) -> np.ndarray:
    file_path = f'./pt/weights_as_npy/{file_path}.npy'
    try:
        weights = np.load(file_path, allow_pickle=True)
        return weights
    except FileNotFoundError:
        raise FileNotFoundError(f"檔案 '{file_path}' 不存在。")
    except Exception as e:
        raise ValueError(f"載入檔案 '{file_path}' 時發生錯誤: {str(e)}")

def Preprocessing(x: np.ndarray) -> np.ndarray:
    x = np.transpose(x, (1, 2, 0))
    H, W, _ = x.shape
    new_H = H - (H % 16)
    new_W = W - (W % 16)
    x_crop = x[:new_H, :new_W, :]
    x_crop = x_crop.astype(np.float32) / 256.0
    mean = np.array([0.62, 0.58, 0.56])
    std = np.array([0.125, 0.125, 0.125])
    x_crop = (x_crop - mean) / std
    result = np.transpose(x_crop, (2, 0, 1))
    
    return result

def post_processing(x: np.ndarray):
    x = np.clip(x, 0, 1)
    x = x * 255.0
    x = x.astype(np.uint8)
    result = np.transpose(x, (1, 2, 0))
    return result

def load_ans_from_txt(file_path: str, shape: tuple) -> np.ndarray:
    file_path = f'./ans/txt/{file_path}.txt'
    try:
        loaded_data = np.loadtxt(file_path, delimiter=',')
        weight_matrix = loaded_data.reshape(shape)
        return weight_matrix
        
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 -> {file_path}")
        return None
    except ValueError:
        # 捕捉 reshape 失敗的錯誤
        elements_in_file = np.loadtxt(file_path, delimiter=',').size
        required_elements = np.prod(shape)
        print(f"錯誤：無法重塑陣列。檔案中有 {elements_in_file} 個元素，但目標形狀 {shape} 需要 {required_elements} 個元素。")
        return None
    
def load_ans_from_npy(file_path: str, shape: tuple) -> np.ndarray:
    file_path = f'./ans/npy/{file_path}.npy'
    try:
        loaded_data = np.load(file_path)
        # 驗證載入的形狀是否與指定形狀匹配
        if loaded_data.shape != shape:
            raise ValueError(f"錯誤：載入陣列形狀 {loaded_data.shape} 與指定形狀 {shape} 不符。")
            
        if shape[0] == 1:
            return np.squeeze(loaded_data, axis=0)
        else:
            return loaded_data
        
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 -> {file_path}")
        return None
    except ValueError as ve:
        print(ve)
        return None
    except Exception as e:
        print(f"錯誤：載入檔案時發生問題 -> {e}")
        return None

def compare_arrays(name: str, arr1: np.ndarray, arr2: np.ndarray, rtol: float = 1e-5, atol: float = 1e-6) -> None:
    if arr1.shape != arr2.shape:
        print(arr1.shape)
        print(arr2.shape)
        raise ValueError("Arrays must have the same shape for comparison.")
    close_mask = np.isclose(arr1, arr2, rtol=rtol, atol=atol)
    num_large_diff = np.sum(~close_mask)
    total_elements = arr1.size
    print(f"{name} : {num_large_diff}/{total_elements}")
    if num_large_diff > 0:
        diff = np.abs(arr1 - arr2)
        rel_diff = diff / (np.abs(arr2) + 1e-10)
        print(f"最大絕對誤差: {np.max(diff)}")
        print(f"最大相對誤差: {np.max(rel_diff)}")
        # 邊緣檢查（32 範圍）
        edge_mask = np.zeros_like(close_mask, dtype=bool)
        if arr1.ndim == 3:
            C, H, W = arr1.shape
            edge_mask[:, :32, :] = True   # 上邊緣
            edge_mask[:, -32:, :] = True  # 下邊緣
            edge_mask[:, :, :32] = True   # 左邊緣
            edge_mask[:, :, -32:] = True  # 右邊緣
            label = "(channel, row, col)"
        elif arr1.ndim == 2:
            H, W = arr1.shape
            edge_mask[:32, :] = True   # 上邊緣
            edge_mask[-32:, :] = True  # 下邊緣
            edge_mask[:, :32] = True   # 左邊緣
            edge_mask[:, -32:] = True  # 右邊緣
            label = "(row, col)"
        elif arr1.ndim == 4:
            C1, C2, H, W = arr1.shape
            edge_mask[:, :, :32, :] = True   # 上邊緣
            edge_mask[:, :, -32:, :] = True  # 下邊緣
            edge_mask[:, :, :, :32] = True   # 左邊緣
            edge_mask[:, :, :, -32:] = True  # 右邊緣
            label = "(dim1, dim2, row, col)"
        else:
            raise ValueError("Array must be 2D, 3D or 4D for edge checking.")
        num_edge_diff = np.sum(~close_mask & edge_mask)
        print(f"邊緣誤差數: {num_edge_diff} ({num_edge_diff / num_large_diff:.2%} of total diff)")
        # 誤差位置範例（前 5 個）
        indices = np.argwhere(~close_mask)[:5]
        print(f"誤差位置範例 {label}:", indices)

def save_image(dehaze, image_name, category):
    """
    保存去霧圖像，使用 NumPy 處理輸入陣列。
    
    參數:
    - dehaze: NumPy 陣列，形狀為 (C, H, W)，其中 C 為通道數，H 為高度，W 為寬度。值域假設為 [0, 1]。
    - image_name: 字串，圖像檔案名稱（例如 'example.jpg'）。
    - category: 字串，結果資料夾類別。
    """
    File_Path = f'./{category}_results'
    if not os.path.exists(File_Path):
        os.makedirs(File_Path)
    
    # 將 CHW 轉換為 HWC 格式
    img_array = np.transpose(dehaze, (1, 2, 0))
    
    # 縮放至 [0, 255] 並轉換為 uint8
    img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
    
    # 創建 PIL 圖像物件並保存
    img = Image.fromarray(img_array)
    save_path = f'{File_Path}/{image_name[:-3]}.png'
    img.save(save_path)