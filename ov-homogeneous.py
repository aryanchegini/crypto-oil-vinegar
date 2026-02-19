import numpy as np
import secrets
from sympy.functions.combinatorial.numbers import totient

N_OIL = 3
N_VINEGAR = 3
TOTAL_VARS = N_OIL + N_VINEGAR
q = 2


def gf_add(a, b):
    return (a + b) % q

def gf_sub(a, b):
    return (a - b + q) % q

def gf_mul(a, b):
    return (a * b) % q

def gf_inv(a):
    # The inverse of a is a^(p-2) mod p for prime p
    if a == 0: return 0
    return pow(int(a), int(totient(q)-1), q)

def gf_mat_mul(A, B):
    """Matrix multiplication modulo P."""
    return np.matmul(A, B) % q

def gf_invert_matrix(M):
    """
    Invert a matrix over a finite field using Gaussian elimination.
    """
    n = M.shape[0]
    M_aug = np.hstack((M, np.eye(n, dtype=int))) # [M | I]
    
    # Forward elimination
    for i in range(n):
        pivot = i
        # Work your way down each column and find the pivot element
        while pivot < n and M_aug[pivot, i] == 0:
            pivot += 1
        # If you reach the bottom of a column and no 1 has been found, you cannot build the identity matrix in the left half of the augmented matrix
        if pivot == n:
            return None # Singular matrix
        
        # Swap rows
        M_aug[[i, pivot]] = M_aug[[pivot, i]]
        
        # Scale row to make pivot 1
        inv_val = gf_inv(M_aug[i, i])
        M_aug[i] = (M_aug[i] * inv_val) % q
        
        # Eliminate other rows
        for j in range(n):
            if i != j:
                factor = M_aug[j, i]
                M_aug[j] = (M_aug[j] - factor * M_aug[i]) % q
    
    # Now, M_aug is of the form [I | M^-1]
    return M_aug[:, n:]

def gf_solve_linear(A, b):
    """Solve Ax = b over finite field using the inverse A^(-1)."""
    A_inv = gf_invert_matrix(A)
    if A_inv is None:
        return None
    return (A_inv @ b) % q


class SecretKey:
    def __init__(self):
        self.S_matrix = None
        self.S_inv_matrix = None
        
        # List of matrices each defining the coefficients for the quadratic terms in each polynomial
        self.central_map = []

class PublicKey:
    def __init__(self):
        # Same structure as central map
        self.equations = []

def generate_keys():
    sk = SecretKey()
    
    # ------Private keys------

    # Affine transformation
    while True:
        # Use secrets module for cryptographic security
        sk.S_matrix = np.array([[secrets.randbelow(q) for _ in range(TOTAL_VARS)] 
                                for _ in range(TOTAL_VARS)], dtype=int)
        
        sk.S_inv_matrix = gf_invert_matrix(sk.S_matrix)
        if sk.S_inv_matrix is not None:
            break


    # Secret central map
    for _ in range(N_OIL):

        A = np.array([[secrets.randbelow(q) for _ in range(TOTAL_VARS)] 
                           for _ in range(TOTAL_VARS)], dtype=int)
        
        # Set Oil x Oil block to zero as there can't be any oil-oil variables
        A[:N_OIL, :N_OIL] = 0
        
        sk.central_map.append(A)
        
    # ------Public quadratic map------
    # P(x) = CentralMap( S * x + v )
    
    pk = PublicKey()
    
    S = sk.S_matrix
    
    for Q_priv in sk.central_map:
        
        # Q_pub = S^T * Q_priv * S
        pk.equations.append((S.T @ Q_priv @ S) % q)
        
    return sk, pk

def sign(sk, message_vector):
    """
    Generate signature for message y (vector of length N_OIL).
    Returns signature x (vector of length TOTAL_VARS).
    """
    attempts = 0
    while True:
        attempts += 1
        if attempts > 100:
            raise RuntimeError("Could not find signature solution after 100 attempts.")
            
        # 1. Pick random Vinegar variables
        vinegar_vars = np.array([secrets.randbelow(q) for _ in range(N_VINEGAR)], dtype=int)
        
        # 2. Build Linear System for Oil variables
        # We need to solve: M * oil = rhs
        M = np.zeros((N_OIL, N_OIL), dtype=int)
        rhs = np.zeros(N_OIL, dtype=int)
        
        for k in range(N_OIL):
            Q = sk.central_map[k]
            y_val = message_vector[k]
            

            # RHS - bottom right quadrant of quadratic terms (vinegar x vinegar) that become constant
            Q_vv = Q[N_OIL:, N_OIL:]
            val_quad_vin = (vinegar_vars @ Q_vv @ vinegar_vars) % q
            rhs[k] = gf_sub(y_val, val_quad_vin)


            # Build the matrix M - coefficients of the oil variables
            
            # Mixed quadratic oil-vin or vin-oil terms (oil * Q_ov * vin  AND  vin * Q_vo * oil)
            Q_ov = Q[:N_OIL, N_OIL:] # top right quadrant
            Q_vo = Q[N_OIL:, :N_OIL] # bottom left quadrant
            oil_vin_coeffs = (Q_ov @ vinegar_vars) % q
            vin_oil_coeffs = (vinegar_vars @ Q_vo) % q
            
            M[k] = (oil_vin_coeffs + vin_oil_coeffs) % q
            
        # Solve for oil
        oil_sol = gf_solve_linear(M, rhs)
        
        if oil_sol is not None:
            internal_vars = np.concatenate((oil_sol, vinegar_vars))
            
            # apply inverse affine on A to find x
            signature = (sk.S_inv_matrix @ internal_vars) % q
            return signature

def verify(pk, message, signature):
    """
    Verify signature against public key equations.
    """
    x = np.array(signature)
    
    for k in range(N_OIL):
        Q = pk.equations[k]
        
        # Evaluate P_k(x)
        val = (x @ Q @ x) % q
        
        if val != message[k]:
            return False
    return True

def save_keys_to_file(sk, pk, filename="ov_keys.txt"):
    with open(filename, "w") as f:
        f.write("---------- OIL AND VINEGAR SCHEME ----------\n")
        f.write(f"Parameters: n={N_OIL}, v={N_VINEGAR}, GF{q}\n\n")
        
        f.write("--- SECRET KEY ---\n")
        f.write("(1) Affine Matrix S (Shape: {})\n".format(sk.S_matrix.shape))
        f.write(str(sk.S_matrix) + "\n\n")
        
        f.write("(3) Central Map (Trapdoor)\n")
        for i, Q_priv in enumerate(sk.central_map):
            f.write("Quad:\n")
            f.write(str(Q_priv) + "\n\n")
            
        f.write("--- PUBLIC KEY ---\n")
        for i, Q_pub in enumerate(pk.equations):
            f.write("Quad:\n")
            f.write(str(Q_pub) + "\n\n")


if __name__ == "__main__":
    print(f"Initializing Oil & Vinegar (N={N_OIL}, V={N_VINEGAR}, GF={q})...")
    
    sk, pk = generate_keys()    
    save_keys_to_file(sk, pk)
    
    msg = np.array([1, 0, 1,1,1])
    print(f"\nMessage to sign: {msg}")
    
    try:
        sig = sign(sk, msg)
        print(f"Signature: {sig}")
        
        is_valid = verify(pk, msg, sig)
        print(f"Verification: {'VALID' if is_valid else 'INVALID'}")
        
        sig[0] = (sig[0] + 1) % q
        is_valid_tamper = verify(pk, msg, sig)
        print(f"Tamper Check: {'VALID' if is_valid_tamper else 'INVALID'}")
        
    except Exception as e:
        print(f"Error: {e}")
