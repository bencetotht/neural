use rand::{thread_rng, ThreadRng};


pub struct Matrix {
    pub rows: usize,
    pub cols: usize, 
    pub data: Vec<Vec<f64>>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix{
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        }
    }

    pub fn random(rows: usize, cols: usize) -> Matrix {
        let mut rng = thread_rng();

        let mut result = Matrix::zeros(rows, cols);

        for i in 0..rows {
            for j in 0..cols {
                result.data[i][j] = rng.gen::<f64>() * 2.0 - 1.0;
            }
        }
    }
}
