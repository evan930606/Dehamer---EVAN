import numpy as np
import inout_test

import numpy as np

def compare_numpy_arrays(output_float, ans_float, file_path, output_binary=None, ans_binary=None, input_float=None, input_binary=None, abs_threshold=1e-10, rel_threshold=1e-5):
    if not isinstance(output_float, np.ndarray) or not isinstance(ans_float, np.ndarray):
        raise ValueError("output_float and ans_float must be NumPy arrays.")
    
    if output_float.shape != ans_float.shape:
        raise ValueError("output_float and ans_float must have the same shape.")
    
    if len(output_float.shape) != 3:
        raise ValueError("Arrays must be 3D (C, H, W).")
    
    # Validate optional arrays if provided
    shape = output_float.shape
    total_elements = np.prod(shape)
    for opt_array, name in [(output_binary, 'output_binary'), (ans_binary, 'ans_binary'), 
                            (input_float, 'input_float'), (input_binary, 'input_binary')]:
        if opt_array is not None:
            if not isinstance(opt_array, np.ndarray) or opt_array.shape != shape:
                raise ValueError(f"{name} must be a NumPy array matching the shape {shape}.")
    
    abs_error = np.abs(output_float - ans_float)
    
    # Compute relative error, handling zeros
    rel_error = np.zeros_like(abs_error)
    nonzero_mask = np.abs(ans_float) != 0
    rel_error[nonzero_mask] = abs_error[nonzero_mask] / np.abs(ans_float[nonzero_mask])
    rel_error[~nonzero_mask] = np.where(abs_error[~nonzero_mask] > 0, np.inf, 0)
    
    # Find indices with significant errors
    mask = (abs_error > abs_threshold) | (rel_error > rel_threshold)
    indices = np.argwhere(mask)
    num_errors = len(indices)
    
    # Determine which columns to include based on non-None inputs
    include_input_binary = input_binary is not None
    include_input_float = input_float is not None
    include_output_binary = output_binary is not None
    include_ans_binary = ans_binary is not None
    
    # Define fixed widths for alignment (adjust as needed based on expected data)
    width_w_h_c = 5  # Right-aligned for indices
    width_binary = 12  # Left-aligned for binary strings (assuming up to 9-10 bits)
    width_float = 25  # Right-aligned for floats in scientific notation
    
    # Build dynamic header with alignment
    header_parts = [
        f"{'C':>{width_w_h_c}}",
        f"{'H':>{width_w_h_c}}",
        f"{'W':>{width_w_h_c}}"
    ]
    format_parts = [
        f"{{:>{width_w_h_c}}}",
        f"{{:>{width_w_h_c}}}",
        f"{{:>{width_w_h_c}}}"
    ]
    if include_input_binary:
        header_parts.append(f"{'input_binary':<{width_binary}}")
        format_parts.append(f"{{:<{width_binary}}}")
    if include_input_float:
        header_parts.append(f"{'input_float':>{width_float}}")
        format_parts.append(f"{{:>{width_float}.18e}}")
    if include_output_binary:
        header_parts.append(f"{'output_binary':<{width_binary}}")
        format_parts.append(f"{{:<{width_binary}}}")
    if include_ans_binary:
        header_parts.append(f"{'ans_binary':<{width_binary}}")
        format_parts.append(f"{{:<{width_binary}}}")
    header_parts.extend([
        f"{'output_float':>{width_float}}",
        f"{'ans_float':>{width_float}}",
        f"{'e_rel':>{width_float}}",
        f"{'e_abs':>{width_float}}"
    ])
    format_parts.extend([
        f"{{:>{width_float}.18e}}",
        f"{{:>{width_float}.18e}}",
        f"{{:>{width_float}.18e}}",
        f"{{:>{width_float}.18e}}"
    ])
    
    header = " | ".join(header_parts)
    row_template = " | ".join(format_parts) + "\n"
    
    with open(file_path, 'w') as f:
        f.write(header + "\n")
        if num_errors == 0:
            f.write("No significant errors found.\n")
        else:
            for idx in indices:
                c, h, w = idx  # Note: indices are (c, h, w), display as C H W
                o_float = output_float[c, h, w]
                a_float = ans_float[c, h, w]
                e_abs = abs_error[c, h, w]
                e_rel = rel_error[c, h, w]
                
                row_data = [c, h, w]  # Display order: C, H, W
                if include_input_binary:
                    row_data.append(input_binary[c, h, w])
                if include_input_float:
                    row_data.append(input_float[c, h, w])
                if include_output_binary:
                    row_data.append(output_binary[c, h, w])
                if include_ans_binary:
                    row_data.append(ans_binary[c, h, w])
                row_data.extend([o_float, a_float, e_rel, e_abs])
                
                f.write(row_template.format(*row_data))
    
    if num_errors == 0:
        print("No significant errors found; empty file written.")
    else:
        print(f"Comparison results for significant errors saved to {file_path}.")
    
    print(f"Number of errors: {num_errors} / Total: {total_elements}")

if __name__ == '__main__' :
    input_binary  = inout_test.txt_to_numpy_array('../data/input_binary_3_480_640.txt', (3, 480, 640), ' ', str)
    output_binary = inout_test.txt_to_numpy_array('../data/output_binary_3_480_640.txt', (3, 480, 640), ' ', str)
    ans_binary    = inout_test.txt_to_numpy_array('../data/ans_binary_3_480_640.txt', (3, 480, 640), ' ', str)
    input_flaot   = inout_test.binary_to_float(input_binary , 8, 0, False)
    output_flaot  = inout_test.binary_to_float(output_binary, 9, 5, True)
    ans_flaot     = inout_test.binary_to_float(ans_binary, 9, 5, True)
    compare_numpy_arrays(output_flaot, ans_flaot, '../data/error.txt', output_binary, ans_binary, input_flaot, input_binary)