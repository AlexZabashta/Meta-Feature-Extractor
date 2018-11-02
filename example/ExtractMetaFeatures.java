import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

import ru.ifmo.ctddev.ml.mfe.Extractor;

public class ExtractMetaFeatures {

    public static void main(String[] args) {
        int n = 128, m = 16;
        double[][] dataset = generateDataset(n, m, new Random(42));

        int f = n / 2; // Number of objects with first class
        int s = n - f; // Number of objects with second class
        int[] classDistribution = { f, s };

        String[] names = Extractor.list();
        int[] metaFeatureIndexes = IntStream.range(0, 23).toArray();

        double[] metaFeatures = Extractor.extract(n, m, dataset, classDistribution, metaFeatureIndexes);

        for (int i = 0; i < metaFeatureIndexes.length; i++) {
            System.out.println(names[metaFeatureIndexes[i]] + "  = " + metaFeatures[i]);
        }
    }

    static double[][] generateDataset(int n, int m, Random random) {
        double[][] data = new double[n][m];
        double[] x = new double[m];

        for (int j = 0; j < m; j++) {
            x[j] = random.nextGaussian();
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                data[i][j] = random.nextGaussian();
            }
        }

        Arrays.sort(data, Comparator.comparingDouble(new ToDoubleFunction<double[]>() {
            @Override
            public double applyAsDouble(double[] a) {
                double sum = 0;
                for (int j = 0; j < m; j++) {
                    sum += x[j] * a[j];
                }
                return sum;
            }
        }));
        return data;
    }
}
