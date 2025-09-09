import numpy as np

def txt_to_numpy_array(file_path, shape, delimiter=' ', dtype=None):
    if not isinstance(shape, tuple) or not all(isinstance(dim, int) and dim > 0 for dim in shape):
        raise ValueError("Shape must be a tuple of positive integers.")
    
    if dtype is None:
        dtype = float  # str
    
    try:
        shape = (shape[1], shape[2], shape[0])
        flat_array = np.loadtxt(file_path, delimiter=delimiter, dtype=dtype)
        expected_size = np.prod(shape)
        if flat_array.size != expected_size:
            raise ValueError(f"Loaded data size ({flat_array.size}) does not match the expected size from shape {shape} ({expected_size}).")
        # (H, W, C) -> (C, H, W)
        array = flat_array.reshape(shape)
        print(array.shape)
        print(f"Array successfully loaded from {file_path} and reshaped to {shape}.")
        array = array.transpose(2, 0, 1)
        return array
    
    except IOError as e:
        raise IOError(f"Error reading file: {e}")
    except ValueError as ve:
        raise ValueError(f"Data parsing or reshaping error: {ve}")

def numpy_array_to_txt(array, file_path, delimiter=' ', fmt=None):
    if not isinstance(array, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    if fmt is None:
        if np.issubdtype(array.dtype, np.number):
            fmt = '%.18e'
        elif np.issubdtype(array.dtype, np.str_) or np.issubdtype(array.dtype, np.object_):
            fmt = '%s'
        else:
            raise TypeError(f"Unsupported array dtype: {array.dtype}. Provide a custom 'fmt'.")
    
    try:
        if array.ndim <= 2:
            # 直接儲存一維或二維陣列
            np.savetxt(file_path, array, delimiter=delimiter, fmt=fmt, header='')
        else: # (C, H, W) -> (H, W, C)
            shape_str = '_' + '_'.join(map(str, array.shape))
            file_path = f"{file_path}{shape_str}.txt"
            array = array.transpose((1, 2, 0))
            flattened_array = array.ravel()
            np.savetxt(file_path, flattened_array, delimiter=delimiter, fmt=fmt)
        
        print(array.shape)
        print(f"Array successfully saved to {file_path}.")

    except IOError as e:
        raise IOError(f"Error saving file: {e}")

def float_to_binary(array, bits=8, frac_bits=0, twos_complement=False):
    if not isinstance(array, np.ndarray) or not np.issubdtype(array.dtype, np.number):
        raise ValueError("Input must be a NumPy array of numbers (integers or floats).")
    
    if bits < 1:
        raise ValueError("Bit length must be at least 1.")
    if frac_bits < 0:
        raise ValueError("Fractional bits must be non-negative.")
    
    scale_factor = 1 << frac_bits
    scaled_array = np.round(array * scale_factor).astype(np.int64)  # Use int64 to handle larger ranges safely
    
    modulus = 1 << bits
    if twos_complement:
        min_value = -(modulus // 2)
        max_value = (modulus // 2) - 1
        if np.any(scaled_array < min_value) or np.any(scaled_array > max_value):
            raise ValueError(f"Overflow or out-of-range detected after scaling; values must be in [{min_value}, {max_value}] for {bits}-bit signed representation.")
        # Adjust negatives for two's complement
        scaled_array = np.where(scaled_array < 0, (scaled_array + modulus) % modulus, scaled_array)
    else:
        min_value = 0
        max_value = modulus - 1
        if np.any(scaled_array < 0):
            raise ValueError("Negative values are not supported in unsigned mode; set twos_complement=True for signed representation.")
        if np.any(scaled_array > max_value):
            raise ValueError(f"Overflow detected after scaling; values must be in [0, {max_value}] for {bits}-bit unsigned representation.")
    
    format_str = f'0{bits}b'
    converter = np.vectorize(lambda x: format(int(x), format_str))
    return converter(scaled_array)

def binary_to_float(binary_array, bits=8, frac_bits=0, twos_complement=False):
    if not isinstance(binary_array, np.ndarray) or binary_array.dtype.kind not in 'SUO':
        raise ValueError("Input must be a NumPy array of strings.")
    
    if bits < 1:
        raise ValueError("Bit length must be at least 1.")
    if frac_bits < 0:
        raise ValueError("Fractional bits must be non-negative.")
    
    def is_valid_binary(s):
        return isinstance(s, str) and len(s) == bits and all(c in '01' for c in s)
    
    if not np.all(np.vectorize(is_valid_binary)(binary_array)):
        raise ValueError(f"All elements must be strings of exactly {bits} binary digits (0 or 1).")
    
    converter = np.vectorize(lambda s: int(s, 2))
    int_array = converter(binary_array)
    
    if twos_complement:
        modulus = 1 << bits
        mask = int_array >= (modulus // 2)
        int_array[mask] -= modulus
    
    scale_factor = 1 << frac_bits
    float_array = int_array.astype(float) / scale_factor
    
    return float_array
