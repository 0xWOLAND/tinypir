from tinygrad import Tensor, dtypes
import random
import numpy as np

class SimplePIR:
    def __init__(self, m, n, q_bits=34, p_bits=16, std_dev=0.5, seed=None):
        self.m = m
        self.n = n
        self.q = 2**q_bits
        self.p = 2**p_bits
        self.std_dev = std_dev
        self.delta = self.q // self.p
        
        if seed is not None:
            random.seed(seed)
            Tensor.manual_seed(seed)
    
    def gen_matrix_a(self, m, n):
        return Tensor.rand(m, n).mul(self.q).cast(dtypes.int64)
    
    def gen_secret(self, n):
        return Tensor.rand(n).mul(self.q).cast(dtypes.int64)
    
    def gen_hint(self, db):
        a = self.gen_matrix_a(self.m, self.n)
        if not isinstance(db, Tensor):
            db = Tensor(db).cast(dtypes.int64)
        hint = (db @ a) % self.q
        return hint, a
    
    def encrypt(self, v, a, s=None):
        if not isinstance(v, Tensor):
            v = Tensor(v).cast(dtypes.int64)
        
        if s is None:
            s = self.gen_secret(self.n)
        
        e = Tensor.normal((self.m,), mean=0., std=self.std_dev)
        e = (e * (self.p // 8)).cast(dtypes.int64) % self.q
        print(f'e: {e.numpy()}')
        
        as_prod = (a.cast(dtypes.int64) @ s.cast(dtypes.int64)) % self.q
        scaled_v = (v.cast(dtypes.int64) * self.delta) % self.q
        query = (as_prod + e + scaled_v) % self.q
        
        return s, query
    
    def generate_query(self, v, a):
        return self.encrypt(v, a)
    
    def process_query(self, db, query):
        if not isinstance(db, Tensor):
            db = Tensor(db).cast(dtypes.int64)
        if not isinstance(query, Tensor):
            query = Tensor(query).cast(dtypes.int64)
        
        result = (db @ query) % self.q
        return result
    
    def recover(self, hint, s, answer):
        if not isinstance(hint, Tensor):
            hint = Tensor(hint).cast(dtypes.int64)
        if not isinstance(s, Tensor):
            s = Tensor(s).cast(dtypes.int64)
        if not isinstance(answer, Tensor):
            answer = Tensor(answer).cast(dtypes.int64)
        
        hint = hint.cast(dtypes.int64)
        s = s.cast(dtypes.int64)
        answer = answer.cast(dtypes.int64)
        
        hint_s = (hint @ s) % self.q
        diff = (answer - hint_s) % self.q
        
        noise_margin = self.p // 16
        raw = ((diff + (self.delta//2) + noise_margin) // self.delta).cast(dtypes.int64)
        
        half_p = self.p // 2
        raw_np = raw.numpy()
        centered = raw_np.copy()
        
        centered = np.where(centered >= half_p, centered - self.p, centered)
        centered = np.where(centered < -half_p, centered + self.p, centered)
        
        return Tensor(centered)