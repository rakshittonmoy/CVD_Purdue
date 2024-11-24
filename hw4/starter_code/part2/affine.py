import torch
from torch import Tensor

def normalize_measurements(D: Tensor) -> tuple[Tensor, Tensor]:
    """
    Normalize the measurement matrix by subtracting the centroid
    of each frame's image points.
    
    Arguments:
    - D: [2M, N] measurement matrix, where M is the number of frames and N is the number of points.

    Returns:
    - D_normalized: [2M, N] normalized measurement matrix.
    - mean: [M, 2] mean values for each frame.
    """
    num_frames = D.shape[0] // 2
    mean = D.view(num_frames, 2, -1).mean(dim=2)  # Compute the mean for each frame
    D_normalized = D.clone()

    for i in range(num_frames):
        D_normalized[2 * i] -= mean[i, 0]
        D_normalized[2 * i + 1] -= mean[i, 1]

    return D_normalized, mean

def normalize_measurements(D: Tensor) -> tuple[Tensor, Tensor]:
    """
    Normalize the measurement matrix by subtracting the centroid
    of each frame's image points.
    
    Arguments:
    - D: [2M, N] measurement matrix, where M is the number of frames and N is the number of points.

    Returns:
    - D_normalized: [2M, N] normalized measurement matrix.
    - mean: [M, 2] mean values for each frame.
    """
    num_frames = D.shape[0] // 2
    mean = D.view(num_frames, 2, -1).mean(dim=2)  # Compute the mean for each frame
    D_normalized = D.clone()

    for i in range(num_frames):
        D_normalized[2 * i] -= mean[i, 0]
        D_normalized[2 * i + 1] -= mean[i, 1]

    return D_normalized, mean


def get_structure_and_motion(
    D: Tensor, 
    k: int = 3
) -> tuple[Tensor, Tensor]:
    """
    Compute the motion matrix (M) and structure matrix (S) using SVD.

    Arguments:
    - D: [2M, N] normalized measurement matrix.
    - k: Number of singular values to retain (default = 3).

    Returns:
    - M: [2M, k] motion matrix.
    - S: [k, N] structure matrix.
    """
    # U, S, Vt = torch.linalg.svd(D, full_matrices=False)  # SVD decomposition
    # U_k = U[:, :k]  # First k columns of U
    # S_k = torch.diag(S[:k])  # Top k singular values as a diagonal matrix
    # V_k = Vt[:k, :]  # First k rows of Vt

    # # M = U_k @ S_k.sqrt()
    # M = U_k @ torch.sqrt(S_k)  # Motion matrix
    # S = torch.sqrt(S_k) @ V_k  # Structure matrix


    # return M, S

    U, S, Vt = torch.linalg.svd(D, full_matrices=False)
    U_k = U[:, :k]
    S_k = torch.diag(S[:k])
    V_k = Vt[:k, :]

    M = U_k @ torch.sqrt(S_k)
    S = torch.sqrt(S_k) @ V_k

    # Normalize rows of M
    num_frames = M.shape[0] // 2
    for i in range(num_frames):
        M[2 * i] /= torch.norm(M[2 * i])
        M[2 * i + 1] /= torch.norm(M[2 * i + 1])

    return M, S


def get_structure_and_motion(D: Tensor, k: int = 3) -> tuple[Tensor, Tensor]:
    """
    Compute the motion matrix (M) and structure matrix (S) using SVD.

    Arguments:
    - D: [2M, N] normalized measurement matrix.
    - k: Number of singular values to retain (default = 3).

    Returns:
    - M: [2M, k] motion matrix.
    - S: [k, N] structure matrix.
    """
    U, S, Vt = torch.linalg.svd(D, full_matrices=False)  # SVD decomposition
    U_k = U[:, :k]  # First k columns of U
    S_k = torch.diag(S[:k])  # Top k singular values as a diagonal matrix
    V_k = Vt[:k, :]  # First k rows of Vt

    M = U_k @ S_k.sqrt()  # Motion matrix
    S = S_k.sqrt() @ V_k  # Structure matrix

    return M, S


def get_Q(M: Tensor) -> Tensor:
    """
    Compute the Q matrix to resolve affine ambiguity.

    Arguments:
    - M: [2M, k] motion matrix.

    Returns:
    - Q: [k, k] affine ambiguity resolution matrix.
    """
    num_frames = M.shape[0] // 2
    A = torch.zeros((3 * num_frames, 6), dtype=M.dtype, device=M.device)
    b = torch.zeros((3 * num_frames,), dtype=M.dtype, device=M.device)

    for i in range(num_frames):
        m1, m2 = M[2 * i], M[2 * i + 1]
        A[3 * i] = torch.tensor([m1[0]**2, 2*m1[0]*m1[1], 2*m1[0]*m1[2], m1[1]**2, 2*m1[1]*m1[2], m1[2]**2])
        A[3 * i + 1] = torch.tensor([m2[0]**2, 2*m2[0]*m1[1], 2*m2[0]*m1[2], m2[1]**2, 2*m2[1]*m2[2], m2[2]**2])
        A[3 * i + 2] = torch.tensor([m1[0]*m2[0], m1[0]*m2[1] + m1[1]*m2[0], m1[0]*m2[2] + m1[2]*m2[0],
                                     m1[1]*m2[1], m1[1]*m2[2] + m1[2]*m2[1], m1[2]*m2[2]])

        b[3 * i] = 1
        b[3 * i + 1] = 1
        b[3 * i + 2] = 0

    # Solve the least squares problem using torch.linalg.lstsq
    result = torch.linalg.lstsq(A, b)
    l = result.solution[:6]  # Extract the first 6 elements of the solution

    # Form the symmetric matrix L
    L = torch.tensor([[l[0], l[1], l[2]],
                      [l[1], l[3], l[4]],
                      [l[2], l[4], l[5]]])

    # Perform Cholesky decomposition to find Q
    Q = torch.linalg.cholesky(L)
    return Q

def get_Q(M: Tensor) -> Tensor:
    num_frames = M.shape[0] // 2
    A = torch.zeros((3 * num_frames, 6), dtype=M.dtype, device=M.device)
    b = torch.ones((3 * num_frames,), dtype=M.dtype, device=M.device)

    for i in range(num_frames):
        m1, m2 = M[2 * i], M[2 * i + 1]
        A[3 * i] = torch.tensor([m1[0]**2, 2*m1[0]*m1[1], 2*m1[0]*m1[2], m1[1]**2, 2*m1[1]*m1[2], m1[2]**2])
        A[3 * i + 1] = torch.tensor([m2[0]**2, 2*m2[0]*m1[1], 2*m2[0]*m1[2], m2[1]**2, 2*m2[1]*m2[2], m2[2]**2])
        A[3 * i + 2] = torch.tensor([m1[0]*m2[0], m1[0]*m2[1] + m1[1]*m2[0], m1[0]*m2[2] + m1[2]*m2[0],
                                     m1[1]*m2[1], m1[1]*m2[2] + m1[2]*m1[1], m1[2]*m2[2]])

    # Add regularization to A
    lambda_reg = 1e-3
    regularization = torch.eye(A.shape[1], device=A.device) * lambda_reg
    A_reg = torch.cat([A, regularization], dim=0)
    b_reg = torch.cat([b, torch.zeros(A.shape[1], device=A.device)])

    # Solve least squares
    result = torch.linalg.lstsq(A_reg, b_reg)
    l = result.solution[:6]
    print("Solution vector l:", l)

    # Construct symmetric L
    L = torch.tensor([[l[0], l[1], l[2]],
                      [l[1], l[3], l[4]],
                      [l[2], l[4], l[5]]])
    L = (L + L.T) / 2
    print("Matrix L before clamping:", L)

    # Clamp eigenvalues to ensure positive definiteness
    eigenvalues, eigenvectors = torch.linalg.eigh(L)
    print("Eigenvalues of L before clamping:", eigenvalues)
    eigenvalues = torch.clamp(eigenvalues, min=1e-6)
    L = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
    print("Eigenvalues of L after clamping:", eigenvalues)

    # Compute Q using Cholesky
    Q = torch.linalg.cholesky(L)
    print("Matrix Q:", Q)

    return Q


def get_Q(M: Tensor) -> Tensor:
    """
    Compute the Q matrix to resolve affine ambiguity.

    Arguments:
    - M: [2M, k] motion matrix.

    Returns:
    - Q: [k, k] affine ambiguity resolution matrix.
    """
    num_frames = M.shape[0] // 2
    A = torch.zeros((3 * num_frames, 6), dtype=M.dtype, device=M.device)
    b = torch.zeros((3 * num_frames,), dtype=M.dtype, device=M.device)

    for i in range(num_frames):
        m1, m2 = M[2 * i], M[2 * i + 1]
        A[3 * i] = torch.tensor([m1[0]**2, 2*m1[0]*m1[1], 2*m1[0]*m1[2], m1[1]**2, 2*m1[1]*m1[2], m1[2]**2])
        A[3 * i + 1] = torch.tensor([m2[0]**2, 2*m2[0]*m1[1], 2*m2[0]*m1[2], m2[1]**2, 2*m2[1]*m2[2], m2[2]**2])
        A[3 * i + 2] = torch.tensor([m1[0]*m2[0], m1[0]*m2[1] + m1[1]*m2[0], m1[0]*m2[2] + m1[2]*m2[0],
                                     m1[1]*m2[1], m1[1]*m2[2] + m1[2]*m2[1], m1[2]*m2[2]])

        b[3 * i] = 1
        b[3 * i + 1] = 1
        b[3 * i + 2] = 0

    # Solve the least squares problem using torch.linalg.lstsq
    result = torch.linalg.lstsq(A, b)
    l = result.solution[:6]  # Extract the first 6 elements of the solution

    # Form the symmetric matrix L
    L = torch.tensor([[l[0], l[1], l[2]],
                      [l[1], l[3], l[4]],
                      [l[2], l[4], l[5]]])

    # Perform Cholesky decomposition to find Q
    Q = torch.linalg.cholesky(L)
    return Q


def verify_reconstruction(D: Tensor, M: Tensor, S: Tensor) -> float:
    """
    Verify the reconstruction of D using M and S.

    Arguments:
    - D: [2M, N] original measurement matrix.
    - M: [2M, k] motion matrix.
    - S: [k, N] structure matrix.

    Returns:
    - reconstruction_error: Frobenius norm error between D and reconstructed D.
    """
    D_reconstructed = M @ S  # Reconstruct D
    reconstruction_error = torch.norm(D - D_reconstructed, p='fro') / torch.norm(D, p='fro')
    return reconstruction_error.item()


def verify_orthogonality(M: Tensor) -> bool:
    """
    Verify the orthonormality of rows of M for each frame.

    Arguments:
    - M: [2M, k] motion matrix.

    Returns:
    - is_orthonormal: True if all frame submatrices satisfy orthogonality.
    """
    num_frames = M.shape[0] // 2
    for i in range(num_frames):
        m1 = M[2 * i]
        m2 = M[2 * i + 1]
        dot_product = torch.dot(m1, m2).abs()
        norm_m1 = torch.norm(m1)
        norm_m2 = torch.norm(m2)
        if dot_product > 1e-6 or abs(norm_m1 - 1) > 1e-6 or abs(norm_m2 - 1) > 1e-6:
            print(f"Frame {i}: Orthonormality Failed")
            print(f"Dot product: {dot_product}, Norm m1: {norm_m1}, Norm m2: {norm_m2}")
            return False
    return True

def verify_Q(M: Tensor, Q: Tensor) -> bool:
    """
    Verify if Q resolves the orthonormality constraint.

    Arguments:
    - M: [2M, k] motion matrix.
    - Q: [k, k] ambiguity resolution matrix.

    Returns:
    - is_valid: True if Q satisfies the orthonormality constraint.
    """
    MQ = M @ Q
    num_frames = MQ.shape[0] // 2
    for i in range(num_frames):
        m1 = MQ[2 * i]
        m2 = MQ[2 * i + 1]
        dot_product = torch.dot(m1, m2).abs()
        norm_m1 = torch.norm(m1)
        norm_m2 = torch.norm(m2)
        if dot_product > 1e-6 or abs(norm_m1 - 1) > 1e-6 or abs(norm_m2 - 1) > 1e-6:
            return False
    return True



