class NF:    
    @staticmethod
    def _fwd(x, f_z, f_n, steps):
        return NF.map_fwd(x, x, f_z, f_n, steps)
    
    @staticmethod
    def _rev(z, n, f_z, f_n, steps):
        return NF.map_rev(z, n, f_z, f_n, steps)[0] # return x
        
    @staticmethod
    def map_fwd(a, b, f_b2a, f_a2b, steps):
        for _ in range(steps):
            b = f_a2b(a) + b
            a = f_b2a(b) + a
        return a, b
        
    @staticmethod
    def map_bwd(a, b, f_b2a, f_a2b, steps):
        for _ in range(steps):
            a = a - f_b2a(b)
            b = b - f_a2b(a)
        return a, b    