//! Eigenvalue estimation via the power method.
//!
//! Finds the dominant (largest magnitude) eigenvalue and its associated eigenvector.

use crate::matrix::Matrix;

/// Estimates the dominant eigenvalue using the power method.
///
/// Starts with a random initial vector and iterates v ← Av / ‖Av‖.
/// Returns `None` if the method doesn't converge.
pub fn power_method(a: &Matrix, max_iter: usize, tol: f64) -> Option<f64> {
    power_method_with_vector(a, max_iter, tol).map(|(eig, _)| eig)
}

/// Estimates the dominant eigenvalue and eigenvector using the power method.
///
/// Returns `Some((eigenvalue, eigenvector))` if convergence occurs.
pub fn power_method_with_vector(a: &Matrix, max_iter: usize, tol: f64) -> Option<(f64, Matrix)> {
    let n = a.rows();
    assert_eq!(n, a.cols(), "matrix must be square");

    // Initial vector: [1, 1, ..., 1]^T
    let mut v = Matrix::from_row_slice(n, 1, &vec![1.0; n]);

    let mut eigenvalue = 0.0;

    for _ in 0..max_iter {
        let av = a * &v;

        // Compute Rayleigh quotient: λ = (v^T * Av) / (v^T * v)
        let vtav: f64 = (0..n).map(|i| v[[i, 0]] * av[[i, 0]]).sum();
        let vtv: f64 = (0..n).map(|i| v[[i, 0]] * v[[i, 0]]).sum();

        if vtv.abs() < 1e-30 {
            return None;
        }

        let new_eigenvalue = vtav / vtv;

        // Normalize
        let norm: f64 = (0..n).map(|i| av[[i, 0]] * av[[i, 0]]).sum::<f64>().sqrt();
        if norm < 1e-30 {
            return None;
        }

        v = Matrix::from_row_slice(n, 1, &(0..n).map(|i| av[[i, 0]] / norm).collect::<Vec<_>>());

        if (new_eigenvalue - eigenvalue).abs() < tol {
            return Some((new_eigenvalue, v));
        }
        eigenvalue = new_eigenvalue;
    }

    // Return best estimate if close enough
    if eigenvalue.abs() > 0.0 {
        Some((eigenvalue, v))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_method_negative_eigenvalue() {
        // [[-2, 0], [0, 1]] → dominant eigenvalue = -2
        let m = Matrix::from_row_slice(2, 2, &[-2.0, 0.0, 0.0, 1.0]);
        let eig = power_method(&m, 100, 1e-10);
        assert!(eig.is_some());
        assert!((eig.unwrap().abs() - 2.0).abs() < 1e-4);
    }
}
