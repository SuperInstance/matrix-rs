//! QR decomposition via Gram-Schmidt orthogonalization.
//!
//! Decomposes matrix A into A = QR where:
//! - Q is orthogonal (Q^T Q = I)
//! - R is upper triangular

use crate::matrix::Matrix;

/// Result of QR decomposition.
pub struct QrResult {
    q: Matrix,
    r: Matrix,
}

impl QrResult {
    /// Returns the orthogonal matrix Q.
    pub fn q(&self) -> &Matrix { &self.q }
    /// Returns the upper triangular matrix R.
    pub fn r(&self) -> &Matrix { &self.r }
}

/// Performs QR decomposition using modified Gram-Schmidt.
///
/// Works for m×n matrices where m ≥ n.
pub fn qr_decompose(a: &Matrix) -> QrResult {
    let m = a.rows();
    let n = a.cols();
    assert!(m >= n, "requires m >= n for QR decomposition");

    let mut q = Matrix::zeros(m, n);
    let mut r = Matrix::zeros(n, n);

    for j in 0..n {
        // Get column j of A
        let mut v: Vec<f64> = a.column(j);

        // Subtract projections onto previous q vectors
        for i in 0..j {
            let qi: Vec<f64> = (0..m).map(|r| q[[r, i]]).collect();
            let dot: f64 = v.iter().zip(&qi).map(|(&a, &b)| a * b).sum();
            r[[i, j]] = dot;
            for k in 0..m {
                v[k] -= dot * qi[k];
            }
        }

        // Normalize
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        r[[j, j]] = norm;

        if norm > 1e-14 {
            for k in 0..m {
                q[[k, j]] = v[k] / norm;
            }
        }
    }

    QrResult { q, r }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qr_column_vector() {
        let a = Matrix::from_row_slice(3, 1, &[1.0, 2.0, 3.0]);
        let result = qr_decompose(&a);
        // Q should have unit norm columns
        let col = result.q().column(0);
        let norm: f64 = col.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-8);
        // R should be [[||v||]]
        let expected_r = (1.0_f64 + 4.0 + 9.0).sqrt();
        assert!((result.r()[[0, 0]] - expected_r).abs() < 1e-8);
    }
}
