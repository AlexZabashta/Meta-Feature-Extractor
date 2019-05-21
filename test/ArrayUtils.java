
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Locale;

public class ArrayUtils {

    public static double[][] copy(double[][] array) {
        int length = array.length;
        double[][] clone = new double[length][];
        for (int i = 0; i < length; i++) {
            clone[i] = array[i].clone();
        }
        return clone;
    }

    public static double[] copy(int n, double[] array) {
        double[] copy = new double[n];
        System.arraycopy(array, 0, copy, 0, n);
        return copy;
    }

    public static double[][] copy(int n, int m, double[][] matrix) {
        double[][] copy = new double[n][m];
        for (int i = 0; i < n; i++) {
            copy[i] = copy(m, matrix[i]);
        }
        return copy;
    }

    public static int[][] copy(int[][] array) {
        int length = array.length;
        int[][] clone = new int[length][];
        for (int i = 0; i < length; i++) {
            clone[i] = array[i].clone();
        }
        return clone;
    }

    public static int max(int[] array) {
        int max = Integer.MIN_VALUE;
        for (int value : array) {
            max = Math.max(max, value);
        }
        return max;
    }

    public static void print(PrintWriter writer, double[] array) {
        for (int i = 0; i < array.length; i++) {
            if (i != 0) {
                writer.print(' ');
            }
            writer.print(array[i]);
        }
        writer.println();
    }

    public static void print(PrintWriter writer, double[][] matrix) {
        for (double[] array : matrix) {
            print(writer, array);
        }
    }

    public static double[][] merge(double[][]... arrays) {
        int n = arrays.length;
        int length = 0;

        for (double[][] array : arrays) {
            length += array.length;
        }

        double[][] union = new double[length][];

        for (int p = 0, i = 0; i < n; i++) {
            double[][] array = arrays[i];
            for (int j = 0; j < array.length; j++, p++) {
                union[p] = array[j];
            }
        }

        return union;
    }

    public static int[] order(int[] array) {
        int n = array.length;
        Integer[] order = new Integer[n];

        for (int i = 0; i < n; i++) {
            order[i] = i;
        }

        Arrays.sort(order, new Comparator<Integer>() {
            @Override
            public int compare(Integer i, Integer j) {
                return Integer.compare(array[j], array[i]);
            }
        });

        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = order[i];
        }
        return p;
    }

    public static void print(double[] array) {
        for (double s : array) {
            System.out.printf(Locale.ENGLISH, "%7.3f ", s);
        }
    }

    public static void print(double[][] matrix) {
        for (double[] array : matrix) {
            print(array);
            System.out.println();
        }
        System.out.println();
    }

}
