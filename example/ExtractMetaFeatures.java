import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;
import java.util.function.ToDoubleFunction;
import java.util.logging.LogManager;

import ru.ifmo.ctddev.ml.mfe.Landmarks;
import ru.ifmo.ctddev.ml.mfe.MetaFeatures;
import ru.ifmo.ctddev.ml.mfe.Utils;
import weka.core.Instances;

public class ExtractMetaFeatures {

    public static void main(String[] args) {
        LogManager.getLogManager().reset();
        final int objects = 128, features = 16, classes = 2;

        double[][] data = generateDataset(objects, features, new Random(42));

        int[] labels = new int[objects];
        for (int i = 0; i < objects; i++) {
            if (i * 2 < objects) {
                labels[i] = 0;
            } else {
                labels[i] = 1;
            }
        }

        long start = System.currentTimeMillis();
        Instances instances = Utils.convert(objects, features, classes, data, labels);

        double[] metaFeatures = MetaFeatures.extract(objects, features, classes, data, labels, instances);
        for (int i = 0; i < metaFeatures.length; i++) {
            System.out.println(MetaFeatures.name(i) + " = " + metaFeatures[i]);
        }

        double[] landmarks = Landmarks.extract(instances);

        for (int i = 0; i < landmarks.length; i++) {
            System.out.println(Landmarks.name(i) + " = " + landmarks[i]);
        }
        long finish = System.currentTimeMillis();

        System.out.println(finish - start);

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
