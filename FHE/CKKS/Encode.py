##
## Credit: https://blog.openmined.org/ckks-explained-part-1-simple-encoding-and-decoding/
##

import numpy as np
from numpy.polynomial import Polynomial

class CKKSEncoder:
    """Basic CKKS encoder to encode complex vectors into polynomials."""
    
    def __init__(self, M: int):
        """Initialization of the encoder for M a power of 2. 
        
        xi, which is an M-th root of unity will, be used as a basis for our computations.
        """
        self.xi = np.exp(2 * np.pi * 1j / M)
        self.M = M
        
    # DFZ: Should we really list xi as a parameter? (It can be calculated from M)
    @staticmethod
    def vandermonde(xi: np.complex128, M: int) -> np.array:
        """Computes the Vandermonde matrix from a m-th root of unity."""
        
        N = M // 2
        matrix = []
        # We will generate each row of the matrix
        for i in range(N):
            # For each row we select a different root
            root = xi ** (2 * i + 1)
            row = []

            # Then we store its powers
            for j in range(N):
                row.append(root ** j)
            matrix.append(row)
        return matrix
    
    def sigma_inverse(self, b: np.array) -> Polynomial:
        """Encodes the vector b in a polynomial using an M-th root of unity."""

        # First we create the Vandermonde matrix
        A = CKKSEncoder.vandermonde(self.xi, self.M)

        # Then we solve the system
        coeffs = np.linalg.solve(A, b)

        # Finally we output the polynomial
        p = Polynomial(coeffs)
        return p

    def sigma(self, p: Polynomial) -> np.array:
        """Decodes a polynomial by applying it to the M-th roots of unity."""

        outputs = []
        N = self.M // 2

        # We simply apply the polynomial on the roots
        for i in range(N):
            root = self.xi ** (2 * i + 1)
            output = p(root)  ## DFZ: So succinct?
            outputs.append(output)
        return np.array(outputs)
    
##
## Test
##

# Parameters
M = 8
N = M // 2
xi = np.exp(2 * np.pi * 1j / M)
print(f'xi = {xi}')

# Encoder object
encoder = CKKSEncoder(M)

# Single plaintext
b = np.array([1, 2, 3, 4])
print(f'b = {b}')
p = encoder.sigma_inverse(b)
print(f'p = {p}')
b_reconstructed = encoder.sigma(p)
print(f'b_reconstructed = {b_reconstructed}')

# Homomorphic add
m1 = np.array([1, 2, 3, 4])
m2 = np.array([1, -2, 3, -4])
p1 = encoder.sigma_inverse(m1)
p2 = encoder.sigma_inverse(m2)
p_add = p1 + p2
print(f'p_add = {p_add}')
m_add_recovered = encoder.sigma(p_add)
print(f'm_add_recovered = {m_add_recovered}')

# Homomorphic multiplication
poly_modulo = Polynomial([1,0,0,0,1])
print(f'poly_modulo = {poly_modulo}')
p_mult = p1 * p2 % poly_modulo
m_mult_recover = encoder.sigma(p_mult)
print(f'm_mult_recover = {m_mult_recover}')

