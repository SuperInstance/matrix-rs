//! Core [`Matrix`] type and basic operations.

use std::fmt;
use std::ops::{Add, Sub, Mul, Index, IndexMut};

/// A dense matrix of `f64` values stored in row-major order.
#[derive(Clone, Debug)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    /// Creates a matrix from a row-major slice.
    ///
    /// # Panics
    /// Panics if `data.len() != rows * cols`.
    pub fn from_row_slice(rows: usize, cols: usize, data: &[f64]) -> Self {
        assert_eq!(data.len(), rows * cols, "data length mismatch");
        Matrix { rows, cols, data: data.to_vec() }
    }

    /// Creates a zero matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix { rows, cols, data: vec![0.0; rows * cols] }
    }

    /// Creates an identity matrix.
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m[[i, i]] = 1.0;
        }
        m
    }

    /// Returns the number of rows.
    pub fn rows(&self) -> usize { self.rows }

    /// Returns the number of columns.
    pub fn cols(&self) -> usize { self.cols }

    /// Returns a reference to the underlying data slice.
    pub fn as_slice(&self) -> &[f64] { &self.data }

    /// Scales all elements by a factor.
    pub fn scale(&self, factor: f64) -> Self {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|&x| x * factor).collect(),
        }
    }

    /// Returns the transpose.
    pub fn transpose(&self) -> Self {
        let mut result = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[[j, i]] = self[[i, j]];
            }
        }
        result
    }

    /// Returns a column as a vector.
    pub fn column(&self, j: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self[[i, j]]).collect()
    }

    /// Sets a column from a vector.
    pub fn set_column(&mut self, j: usize, v: &[f64]) {
        for i in 0..self.rows {
            self[[i, j]] = v[i];
        }
    }
}

impl Index<[usize; 2]> for Matrix {
    type Output = f64;
    fn index(&self, [r, c]: [usize; 2]) -> &f64 {
        &self.data[r * self.cols + c]
    }
}

impl IndexMut<[usize; 2]> for Matrix {
    fn index_mut(&mut self, [r, c]: [usize; 2]) -> &mut f64 {
        &mut self.data[r * self.cols + c]
    }
}

impl Add for &Matrix {
    type Output = Matrix;
    fn add(self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self.data.iter().zip(&other.data).map(|(&a, &b)| a + b).collect();
        Matrix { rows: self.rows, cols: self.cols, data }
    }
}

impl Sub for &Matrix {
    type Output = Matrix;
    fn sub(self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self.data.iter().zip(&other.data).map(|(&a, &b)| a - b).collect();
        Matrix { rows: self.rows, cols: self.cols, data }
    }
}

impl Mul for &Matrix {
    type Output = Matrix;
    fn mul(self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "matrix dimensions don't match for multiplication");
        let mut result = Matrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self[[i, k]] * other[[k, j]];
                }
                result[[i, j]] = sum;
            }
        }
        result
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.rows {
            for j in 0..self.cols {
                write!(f, "{:10.4}", self[[i, j]])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}
