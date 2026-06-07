//! # matrix-rs
//!
//! Dense matrix operations library implementing LU and QR decomposition,
//! determinant computation, matrix inverse, and eigenvalue estimation.
//!
//! ## Modules
//! - [`matrix`] - Core `Matrix` type and basic operations
//! - [`lu`] - LU decomposition with partial pivoting
//! - [`det`] - Determinant computation
//! - [`inverse`] - Matrix inverse via Gauss-Jordan elimination
//! - [`qr`] - QR decomposition via Gram-Schmidt
//! - [`eigen`] - Eigenvalue estimation via power method

pub mod det;
pub mod eigen;
pub mod inverse;
pub mod lu;
pub mod matrix;
pub mod qr;

pub use matrix::Matrix;

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn mat_approx_eq(a: &Matrix, b: &Matrix, tol: f64) -> bool {
        if a.rows() != b.rows() || a.cols() != b.cols() {
            return false;
        }
        for i in 0..a.rows() {
            for j in 0..a.cols() {
                if !approx_eq(a[[i, j]], b[[i, j]], tol) {
                    return false;
                }
            }
        }
        true
    }

    fn identity(n: usize) -> Matrix {
        Matrix::identity(n)
    }

    // ── Matrix basics ─────────────────────────────────────────

    #[test]
    fn test_create_matrix() {
        let m = Matrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert!((m[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((m[[1, 2]] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_identity() {
        let i = identity(3);
        assert!((i[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((i[[1, 1]] - 1.0).abs() < 1e-10);
        assert!((i[[0, 1]]).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_add() {
        let a = Matrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_row_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);
        let sum = &a + &b;
        assert!(mat_approx_eq(&sum, &Matrix::from_row_slice(2, 2, &[6.0, 8.0, 10.0, 12.0]), 1e-10));
    }

    #[test]
    fn test_matrix_sub() {
        let a = Matrix::from_row_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);
        let b = Matrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let diff = &a - &b;
        assert!(mat_approx_eq(&diff, &Matrix::from_row_slice(2, 2, &[4.0, 4.0, 4.0, 4.0]), 1e-10));
    }

    #[test]
    fn test_matrix_mul() {
        let a = Matrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::from_row_slice(3, 2, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = &a * &b;
        assert_eq!(c.rows(), 2);
        assert_eq!(c.cols(), 2);
        assert!(mat_approx_eq(&c, &Matrix::from_row_slice(2, 2, &[58.0, 64.0, 139.0, 154.0]), 1e-10));
    }

    #[test]
    fn test_mul_identity() {
        let a = Matrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let i = identity(2);
        assert!(mat_approx_eq(&(&a * &i), &a, 1e-10));
        assert!(mat_approx_eq(&(&i * &a), &a, 1e-10));
    }

    #[test]
    fn test_scale() {
        let a = Matrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let scaled = a.scale(2.0);
        assert!(mat_approx_eq(&scaled, &Matrix::from_row_slice(2, 2, &[2.0, 4.0, 6.0, 8.0]), 1e-10));
    }

    #[test]
    fn test_transpose() {
        let a = Matrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = a.transpose();
        assert_eq!(t.rows(), 3);
        assert_eq!(t.cols(), 2);
        assert!((t[[0, 1]] - 4.0).abs() < 1e-10);
    }

    // ── LU decomposition ──────────────────────────────────────

    #[test]
    fn test_lu_simple() {
        let a = Matrix::from_row_slice(2, 2, &[2.0, 1.0, 6.0, 4.0]);
        let result = lu::lu_decompose(&a).unwrap();
        let reconstructed = result.l() * result.u();
        // With permutation: PA = LU, so A = P^T * L * U
        let pa = result.p() * &a;
        assert!(mat_approx_eq(&pa, &reconstructed, 1e-8));
    }

    #[test]
    fn test_lu_3x3() {
        let a = Matrix::from_row_slice(3, 3, &[2.0, 1.0, 1.0, 4.0, 3.0, 3.0, 8.0, 7.0, 9.0]);
        let result = lu::lu_decompose(&a).unwrap();
        let reconstructed = result.l() * result.u();
        let pa = result.p() * &a;
        assert!(mat_approx_eq(&pa, &reconstructed, 1e-8));
    }

    #[test]
    fn test_lu_identity() {
        let i = identity(3);
        let result = lu::lu_decompose(&i).unwrap();
        assert!(mat_approx_eq(&result.l(), &identity(3), 1e-8));
        assert!(mat_approx_eq(&result.u(), &identity(3), 1e-8));
    }

    // ── Determinant ───────────────────────────────────────────

    #[test]
    fn test_det_identity() {
        assert!(approx_eq(det::determinant(&identity(3)), 1.0, 1e-10));
    }

    #[test]
    fn test_det_2x2() {
        let m = Matrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert!(approx_eq(det::determinant(&m), -2.0, 1e-8));
    }

    #[test]
    fn test_det_3x3() {
        let m = Matrix::from_row_slice(3, 3, &[6.0, 1.0, 1.0, 4.0, -2.0, 5.0, 2.0, 8.0, 7.0]);
        // det = 6(-14-40) - 1(28-10) + 1(32+4) = 6(-54) - 18 + 36 = -324 - 18 + 36 = -306
        assert!(approx_eq(det::determinant(&m), -306.0, 1e-6));
    }

    #[test]
    fn test_det_singular() {
        let m = Matrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
        assert!(approx_eq(det::determinant(&m), 0.0, 1e-8));
    }

    #[test]
    fn test_det_upper_triangular() {
        let m = Matrix::from_row_slice(3, 3, &[2.0, 3.0, 1.0, 0.0, 5.0, 4.0, 0.0, 0.0, 3.0]);
        assert!(approx_eq(det::determinant(&m), 30.0, 1e-8));
    }

    // ── Inverse ───────────────────────────────────────────────

    #[test]
    fn test_inverse_2x2() {
        let m = Matrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let inv = inverse::inverse(&m).unwrap();
        let product = &m * &inv;
        assert!(mat_approx_eq(&product, &identity(2), 1e-8));
    }

    #[test]
    fn test_inverse_3x3() {
        let m = Matrix::from_row_slice(3, 3, &[1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0]);
        let inv = inverse::inverse(&m).unwrap();
        let product = &m * &inv;
        assert!(mat_approx_eq(&product, &identity(3), 1e-6));
    }

    #[test]
    fn test_inverse_identity() {
        let inv = inverse::inverse(&identity(3)).unwrap();
        assert!(mat_approx_eq(&inv, &identity(3), 1e-10));
    }

    #[test]
    fn test_inverse_singular() {
        let m = Matrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
        assert!(inverse::inverse(&m).is_none());
    }

    #[test]
    fn test_inverse_roundtrip() {
        let m = Matrix::from_row_slice(3, 3, &[4.0, 3.0, 2.0, 1.0, 5.0, 3.0, 2.0, 1.0, 6.0]);
        let inv = inverse::inverse(&m).unwrap();
        let inv_inv = inverse::inverse(&inv).unwrap();
        assert!(mat_approx_eq(&inv_inv, &m, 1e-4));
    }

    // ── QR decomposition ──────────────────────────────────────

    #[test]
    fn test_qr_simple() {
        let a = Matrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let result = qr::qr_decompose(&a);
        // Q should be orthogonal: Q^T * Q = I
        let qtq = &result.q().transpose() * result.q();
        assert!(mat_approx_eq(&qtq, &identity(2), 1e-8));
        // A = Q * R
        let qr = result.q() * result.r();
        assert!(mat_approx_eq(&qr, &a, 1e-8));
    }

    #[test]
    fn test_qr_3x3() {
        let a = Matrix::from_row_slice(3, 3, &[12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0]);
        let result = qr::qr_decompose(&a);
        let qtq = &result.q().transpose() * result.q();
        assert!(mat_approx_eq(&qtq, &identity(3), 1e-6));
        let qr = result.q() * result.r();
        assert!(mat_approx_eq(&qr, &a, 1e-6));
    }

    #[test]
    fn test_qr_upper_triangular() {
        let a = Matrix::from_row_slice(3, 3, &[2.0, 3.0, 1.0, 0.0, 5.0, 4.0, 0.0, 0.0, 3.0]);
        let result = qr::qr_decompose(&a);
        // R should be upper triangular
        for i in 0..3 {
            for j in 0..i {
                assert!(result.r()[[i, j]].abs() < 1e-8, "R[{}, {}] = {} not zero", i, j, result.r()[[i, j]]);
            }
        }
    }

    #[test]
    fn test_qr_identity() {
        let i = identity(3);
        let result = qr::qr_decompose(&i);
        assert!(mat_approx_eq(result.q(), &identity(3), 1e-8));
        assert!(mat_approx_eq(result.r(), &identity(3), 1e-8));
    }

    // ── Eigenvalue estimation ─────────────────────────────────

    #[test]
    fn test_eigen_dominant_2x2() {
        // [[4, 1], [2, 3]] has eigenvalues 5 and 2
        let m = Matrix::from_row_slice(2, 2, &[4.0, 1.0, 2.0, 3.0]);
        let eig = eigen::power_method(&m, 100, 1e-10);
        assert!(eig.is_some());
        assert!(approx_eq(eig.unwrap(), 5.0, 1e-4));
    }

    #[test]
    fn test_eigen_dominant_3x3() {
        // Diagonal matrix: eigenvalues are 5, 3, 1
        let m = Matrix::from_row_slice(3, 3, &[5.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0]);
        let eig = eigen::power_method(&m, 100, 1e-10);
        assert!(eig.is_some());
        assert!(approx_eq(eig.unwrap(), 5.0, 1e-4));
    }

    #[test]
    fn test_eigen_dominant_negative() {
        // [[-3, 0], [0, 2]] has eigenvalues -3 and 2
        let m = Matrix::from_row_slice(2, 2, &[-3.0, 0.0, 0.0, 2.0]);
        let eig = eigen::power_method(&m, 100, 1e-10);
        assert!(eig.is_some());
        // Dominant eigenvalue is -3 (|−3| > |2|)
        assert!(approx_eq(eig.unwrap().abs(), 3.0, 1e-4));
    }

    #[test]
    fn test_eigen_identity() {
        let i = identity(3);
        let eig = eigen::power_method(&i, 100, 1e-10);
        assert!(eig.is_some());
        assert!(approx_eq(eig.unwrap(), 1.0, 1e-4));
    }

    #[test]
    fn test_eigen_with_eigenvector() {
        let m = Matrix::from_row_slice(2, 2, &[4.0, 1.0, 2.0, 3.0]);
        let result = eigen::power_method_with_vector(&m, 100, 1e-10);
        assert!(result.is_some());
        let (eig, vec) = result.unwrap();
        assert!(approx_eq(eig, 5.0, 1e-4));
        // Verify Av ≈ λv
        let av = &m * &vec;
        let lv = vec.scale(eig);
        assert!(mat_approx_eq(&av, &lv, 1e-4));
    }
}
