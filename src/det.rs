//! Determinant computation via LU decomposition.

use crate::matrix::Matrix;
use crate::lu;

/// Computes the determinant of a square matrix.
///
/// Uses LU decomposition: det(A) = (-1)^swaps * product of diagonal of U.
pub fn determinant(a: &Matrix) -> f64 {
    let n = a.rows();
    assert_eq!(n, a.cols(), "matrix must be square");

    if n == 1 {
        return a[[0, 0]];
    }
    if n == 2 {
        return a[[0, 0]] * a[[1, 1]] - a[[0, 1]] * a[[1, 0]];
    }

    match lu::lu_decompose(a) {
        Some(result) => {
            let mut det = if result.swaps % 2 == 0 { 1.0 } else { -1.0 };
            for i in 0..n {
                det *= result.u()[[i, i]];
            }
            det
        }
        None => 0.0, // Singular
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_det_1x1() {
        let m = Matrix::from_row_slice(1, 1, &[5.0]);
        assert!((determinant(&m) - 5.0).abs() < 1e-10);
    }
}
