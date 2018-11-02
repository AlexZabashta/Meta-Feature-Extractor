package ru.ifmo.ctddev.ml.mfe;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Extractor {

    static final Map<String, Integer> indexes = new HashMap<>();

    // List<E>

    public static final int MAX_METAFEATURES = 123;

    static final String[] names = new String[MAX_METAFEATURES];

    static int[] notEmpty(int[] array) {
        if (array.length == 0) {
            array = new int[MAX_METAFEATURES];
            for (int i = 0; i < MAX_METAFEATURES; i++) {
                array[i] = i;
            }
        }
        return array;
    }

    public static int metaFeatureIndexes(String... metaFeatureNames) {
        int length = metaFeatureNames.length;
        throw new UnsupportedOperationException(); // TODO
    }

    public static double[] extract(int objects, int features, double[][] dataset, int[] c, int... metaFeatureIndexes) {
        metaFeatureIndexes = notEmpty(metaFeatureIndexes);
        int length = metaFeatureIndexes.length;
        double[] metaFeatures = new double[length];

        return metaFeatures;
    }

    public static String[] list() {
        return names.clone();
    }

}
