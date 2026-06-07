//! LU decomposition with partial pivoting.
//!
//! Decomposes matrix A into PA = LU where:
//! - P is a permutation matrix
//! - L is lower triangular with unit diagonal
//! - U is upper triangular

use crate::matrix::Matrix;

/// Result of LU decomposition.
pub struct LuResult {
    l: Matrix,
    u: Matrix,
    p: Matrix,
    /// Number of row swaps (for determinant sign).
    pub swaps: usize,
}

impl LuResult {
    /// Returns the lower triangular matrix L.
    pub fn l(&self) -> &Matrix { &self.l }
    /// Returns the upper triangular matrix U.
    pub fn u(&self) -> &Matrix { &self.u }
    /// Returns the permutation matrix P.
    pub fn p(&self) -> &Matrix { &self.p }
}

/// Performs LU decomposition with partial pivoting.
///
/// Returns `None` if the matrix is singular (zero pivot encountered).
pub fn lu_decompose(a: &Matrix) -> Option<LuResult> {
    let n = a.rows();
    assert_eq!(n, a.cols(), "matrix must be square");

    let mut u = a.clone();
    let mut l = Matrix::identity(n);
    let mut perm: Vec<usize> = (0..n).collect();
    let mut swaps = 0;

    for k in 0..n {
        // Find pivot
        let mut max_val = u[[k, k]].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            if u[[i, k]].abs() > max_val {
                max_val = u[[i, k]].abs();
                max_row = i;
            }
        }

        if max_val < 1e-14 {
            return None; // Singular
        }

        // Swap rows
        if max_row != k {
            for j in 0..n {
                let tmp = u[[k, j]];
                u[[k, j]] = u[[max_row, j]];
                u[[max_row, j]] = tmp;
            }
            for j in 0..k {
                let tmp = l[[k, j]];
                l[[k, j]] = l[[max_row, j]];
                l[[max_row, j]] = tmp;
            }
            perm.swap(k, max_row);
            swaps += 1;
        }

        // Eliminate below
        for i in (k + 1)..n {
            let factor = u[[i, k]] / u[[k, k]];
            l[[i, k]] = factor;
            for j in k..n {
                u[[i, j]] -= factor * u[[k, j]];
            }
        }
    }

    // Build permutation matrix
    let mut p = Matrix::zeros(n, n);
    for (i, &pi) in perm.iter().enumerate() {
        p[[i, pi]] = 1.0;
    }

    Some(LuResult { l, u, p, swaps })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lu_singular() {
        let m = Matrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
        assert!(lu_decompose(&m).is_none());
    }
}
