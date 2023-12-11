##
## Credit: https://blog.openmined.org/ckks-explained-part-1-simple-encoding-and-decoding/
##

import numpy as np
from numpy.polynomial import Polynomial
from fastcore.foundation import patch_to

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
    
@patch_to(CKKSEncoder)
def pi(self, z: np.array) -> np.array:
    """Projects a vector of H into C^{N/2}."""
    
    N = self.M // 4
    return z[:N]

@patch_to(CKKSEncoder)
def pi_inverse(self, z: np.array) -> np.array:
    """Expands a vector of C^{N/2} by expanding it with its complex conjugate."""
    
    z_conjugate = z[::-1]
    z_conjugate = [np.conjugate(x) for x in z_conjugate]
    return np.concatenate([z, z_conjugate])

@patch_to(CKKSEncoder)
def create_sigma_R_basis(self):
    """Creates the basis (sigma(1), sigma(X), ..., sigma(X** N-1))."""

    self.sigma_R_basis = np.array(self.vandermonde(self.xi, self.M)).T
    
@patch_to(CKKSEncoder)
def __init__(self, M):
    """Initialize with the basis"""
    self.xi = np.exp(2 * np.pi * 1j / M)
    self.M = M
    self.create_sigma_R_basis()

@patch_to(CKKSEncoder)
def compute_basis_coordinates(self, z):
    """Computes the coordinates of a vector with respect to the orthogonal lattice basis."""
    output = np.array([np.real(np.vdot(z, b) / np.vdot(b,b)) for b in self.sigma_R_basis])
    return output

def round_coordinates(coordinates):
    """Gives the integral rest."""
    coordinates = coordinates - np.floor(coordinates)
    return coordinates

def coordinate_wise_random_rounding(coordinates):
    """Rounds coordinates randonmly."""
    r = round_coordinates(coordinates)
    f = np.array([np.random.choice([c, c-1], 1, p=[1-c, c]) for c in r]).reshape(-1)
    
    rounded_coordinates = coordinates - f
    rounded_coordinates = [int(coeff) for coeff in rounded_coordinates]
    return rounded_coordinates

@patch_to(CKKSEncoder)
def sigma_R_discretization(self, z):
    """Projects a vector on the lattice using coordinate wise random rounding."""
    coordinates = self.compute_basis_coordinates(z)
    
    rounded_coordinates = coordinate_wise_random_rounding(coordinates)
    y = np.matmul(self.sigma_R_basis.T, rounded_coordinates)
    return y

@patch_to(CKKSEncoder)
def __init__(self, M:int, scale:float):
    """Initializes with scale."""
    self.xi = np.exp(2 * np.pi * 1j / M)
    self.M = M
    self.create_sigma_R_basis()
    self.scale = scale
    
@patch_to(CKKSEncoder)
def encode(self, z: np.array) -> Polynomial:
    """Encodes a vector by expanding it first to H,
    scale it, project it on the lattice of sigma(R), and performs
    sigma inverse.
    """
    pi_z = self.pi_inverse(z)
    scaled_pi_z = self.scale * pi_z
    rounded_scale_pi_zi = self.sigma_R_discretization(scaled_pi_z)
    p = self.sigma_inverse(rounded_scale_pi_zi)
    
    # We round it afterwards due to numerical imprecision
    coef = np.round(np.real(p.coef)).astype(int)
    p = Polynomial(coef)
    return p

@patch_to(CKKSEncoder)
def decode(self, p: Polynomial) -> np.array:
    """Decodes a polynomial by removing the scale, 
    evaluating on the roots, and project it on C^(N/2)"""
    rescaled_p = p / self.scale
    z = self.sigma(rescaled_p)
    pi_z = self.pi(z)
    return pi_z

##
## Test: Full CKKS encoding
##

# Parameters
M = 8

encoder = CKKSEncoder(M, 1)
z = np.array([0 + 1j, 1 + 2j])
z_expand = encoder.pi_inverse(z)
print(f'z_expand = {z_expand}')
print(f'encoder.sigma_R_basis = {encoder.sigma_R_basis}')
print(f'encoder.sigma_R_basis.shape = {encoder.sigma_R_basis.shape}')

# Here we simply take a vector whose coordinates are (1,1,1,1) in the lattice basis
coordinates = [1,1,1,1]
b = np.matmul(encoder.sigma_R_basis.T, coordinates)
print(f'b = {b}')

p = encoder.sigma_inverse(b)
print(f'p = {p}')

scale = 64
encoder = CKKSEncoder(M, scale)
z = np.array([3 +4j, 2 - 1j])
print(f'z = {z}')
p = encoder.encode(z)
print(f'p = {p}')
z_recovery = encoder.decode(p)
print(f'z_recovery = {z_recovery}')

exit(0)

##
## Test: Vanilla encoding
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

