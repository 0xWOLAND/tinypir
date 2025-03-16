from simplepir.pir import SimplePIR
from tinygrad import Tensor, dtypes
import random
import numpy as np

def test_pir():
    matrix_height = 10
    matrix_width = 10
    
    # Database entry size
    db = Tensor.randn(matrix_height, matrix_width).mul(2**8).cast(dtypes.int64)
    
    target_col = random.randint(0, matrix_width-1)
    v = Tensor.zeros(matrix_width, dtype=dtypes.int64).contiguous()
    
    v_list = v.numpy().tolist()
    v_list[target_col] = 1
    v = Tensor(v_list, dtype=dtypes.int64).contiguous()
    
    expected = db[:, target_col].numpy()
    
    # Parameters tuned for better accuracy
    pir = SimplePIR(matrix_height, 2048, q_bits=34, p_bits=16, std_dev=1.5, seed=42)
    hint, a = pir.gen_hint(db)
    s, query = pir.generate_query(v, a)
    answer = pir.process_query(db, query)
    result = pir.recover(hint, s, answer)
    
    print("Expected:", expected)
    print("Result:", result.numpy())
    
    diff = np.abs(expected - result.numpy())
    print("Max difference:", np.max(diff))
    
    tolerance = 50
    is_close = np.all(diff <= tolerance)
    print("Success:", is_close)

if __name__ == "__main__":
    test_pir()