
import java.util.Random;

public class MatrixUtils {

    public static double[][] inv(int n, double[][] matrix) {
        double[][] inverse = new double[n][n];

        double[] swapRow;

        for (int i = 0; i < n; i++) {
            inverse[i][i] = 1.0;
        }

        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }

        int k = n;

        for (int i = 0; i < k; i++) {
            int maxR = i, maxC = i;

            for (int row = i; row < n; row++) {
                for (int col = i; col < n; col++) {
                    if (Math.abs(matrix[row][p[col]]) > Math.abs(matrix[maxR][p[maxC]])) {
                        maxR = row;
                        maxC = col;
                    }
                }
            }

            swapRow = matrix[i];
            matrix[i] = matrix[maxR];
            matrix[maxR] = swapRow;

            swapRow = inverse[i];
            inverse[i] = inverse[maxR];
            inverse[maxR] = swapRow;

            int swapCol = p[i];
            p[i] = p[maxC];
            p[maxC] = swapCol;

            if (Math.abs(matrix[i][p[i]]) < 1e-6) {
                k = i;

                for (int row = k; row < n; row++) {
                    for (int col = 0; col < n; col++) {
                        inverse[row][col] = 0.0;
                    }
                }

                break;
            }

            double inv = 1 / matrix[i][p[i]];

            for (int col = 0; col < n; col++) {
                matrix[i][col] *= inv;
            }
            for (int col = 0; col < n; col++) {
                inverse[i][col] *= inv;
            }

            for (int row = i + 1; row < n; row++) {
                double scale = matrix[row][p[i]];
                for (int col = 0; col < n; col++) {
                    matrix[row][col] -= scale * matrix[i][col];
                }
                for (int col = 0; col < n; col++) {
                    inverse[row][col] -= scale * inverse[i][col];
                }
            }

        }

        for (int i = k - 1; i > 0; i--) {
            for (int row = i - 1; row >= 0; row--) {
                double scale = matrix[row][p[i]];
                for (int col = 0; col < n; col++) {
                    matrix[row][col] -= scale * matrix[i][col];
                }
                for (int col = 0; col < n; col++) {
                    inverse[row][col] -= scale * inverse[i][col];
                }
            }
            // print(n, a, b);
        }

        double[][] c = new double[n][];

        for (int i = 0; i < n; i++) {
            c[p[i]] = inverse[i];
        }

        return c;
    }

    public static void main(String[] args) {
        Random random = new Random();

        int n = 4;

        double[][] matrix = new double[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                matrix[i][j] = random.nextGaussian();
            }
        }

        double[][] inv = inv(n, ArrayUtils.copy(matrix));

        ArrayUtils.print(matrix);
        ArrayUtils.print(inv);

        ArrayUtils.print(mul(matrix, inv));
        ArrayUtils.print(mul(inv, matrix));

    }

    public static double[][] mul(double[][] a, double[][] b) {

        int l = a.length;
        int m = b.length;
        int r = width(b);

        if (width(a) != m) {
            throw new IllegalStateException("matrixes dims not consistent");
        }

        return mul(l, m, r, a, b);
    }

    public static double[][] mul(int n, double[][] a, double[][] b) {
        return mul(n, n, n, a, b);
    }

    public static double[][] mul(int l, int m, int r, double[][] a, double[][] b) {
        double[][] c = new double[l][r];

        for (int i = 0; i < l; i++) {
            for (int j = 0; j < m; j++) {
                for (int k = 0; k < r; k++) {
                    c[i][k] += a[i][j] * b[j][k];
                }
            }
        }

        return c;
    }

    public static double[][] sqrt(double[][] matrix) {
        return sqrt(matrix.length, matrix);
    }

    /**
     * Computes the Cholesky decomposition of the current matrix \f$ matrix \f$.
     * This method is taken from {@link jMEF.PMatrix#Cholesky}.
     * 
     * @author Vincent Garcia
     * @author Frank Nielsen
     * @return a lower triangular matrix
     */
    public static double[][] sqrt(int n, double[][] matrix) {
        double[][] sqrt = new double[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double sum = 0.0d;
                for (int k = 0; k < j; k++) {
                    sum += sqrt[i][k] * sqrt[j][k];
                }
                if (i == j) {
                    sqrt[i][i] = Math.sqrt(matrix[i][i] - sum);
                } else {
                    sqrt[i][j] = (matrix[i][j] - sum) / sqrt[j][j];
                }
            }
            if (sqrt[i][i] <= 0.0d) {
                throw new IllegalStateException("Matrix is not positive definite!");
            }
        }
        return sqrt;
    }

    public static double[][] transpose(double[][] matrix) {
        return transpose(matrix.length, width(matrix), matrix);
    }

    public static double[][] transpose(int n, double[][] matrix) {
        return transpose(n, n, matrix);
    }

    public static double[][] transpose(int n, int m, double[][] matrix) {
        double[][] transpose = new double[m][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                transpose[j][i] = matrix[i][j];
            }
        }

        return transpose;
    }

    public static int width(double[][] matrix) {
        int width = -1;

        for (double[] array : matrix) {
            if (width == -1) {
                width = array.length;
            }

            if (width != array.length) {
                throw new IllegalStateException("matrix not rectangular");
            }
        }

        return Math.max(0, width);
    }
}