//! Matrix inverse via Gauss-Jordan elimination.

use crate::matrix::Matrix;

/// Computes the inverse of a square matrix.
///
/// Returns `None` if the matrix is singular.
pub fn inverse(a: &Matrix) -> Option<Matrix> {
    let n = a.rows();
    assert_eq!(n, a.cols(), "matrix must be square");

    // Augment with identity: [A | I]
    let mut aug = Matrix::zeros(n, 2 * n);
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n + i]] = 1.0;
    }

    // Gauss-Jordan elimination
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            if aug[[row, col]].abs() > max_val {
                max_val = aug[[row, col]].abs();
                max_row = row;
            }
        }

        if max_val < 1e-14 {
            return None; // Singular
        }

        // Swap rows
        if max_row != col {
            for j in 0..2 * n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }

        // Scale pivot row
        let pivot = aug[[col, col]];
        for j in 0..2 * n {
            aug[[col, j]] /= pivot;
        }

        // Eliminate column
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[[row, col]];
            for j in 0..2 * n {
                aug[[row, j]] -= factor * aug[[col, j]];
            }
        }
    }

    // Extract inverse from right half
    let mut inv = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inverse_diagonal() {
        let m = Matrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 4.0]);
        let inv = inverse(&m).unwrap();
        assert!((inv[[0, 0]] - 0.5).abs() < 1e-8);
        assert!((inv[[1, 1]] - 0.25).abs() < 1e-8);
    }
}
