package ru.ifmo.ctddev.ml.mfe;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.IntStream;

import com.ifmo.recommendersystem.metafeatures.MetaFeatureExtractor;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class Extractor {

    private static final Map<String, Integer> indexes = new HashMap<>();
    private static final List<MetaFeatureExtractor> extractors = MetaFeatureExtractorsCollection.all();
    public static final int LENGTH;
    static final String[] names;

    static {
        LENGTH = extractors.size();

        names = new String[LENGTH];
        for (int i = 0; i < LENGTH; i++) {
            MetaFeatureExtractor extractor = extractors.get(i);
            names[i] = extractor.getName();
            indexes.put(names[i], i);
        }

    }

    static int[] notEmpty(int[] array) {
        if (array.length == 0) {
            array = new int[LENGTH];
            for (int i = 0; i < LENGTH; i++) {
                array[i] = i;
            }
        }
        return array;
    }

    public static int[] metaFeatureIndexes(String... metaFeatureNames) {
        int length = metaFeatureNames.length;
        int[] metaFeatureIndexes = new int[length];

        for (int i = 0; i < length; i++) {
            Integer index = indexes.get(metaFeatureNames[i]);
            if (index == null) {
                throw new IllegalArgumentException("Can't find meta-feature name = " + metaFeatureNames[i]);
            }
            metaFeatureIndexes[i] = index;
        }

        return metaFeatureIndexes;
    }

    public static double[] extract(int objects, int features, double[][] dataset, int[] classDistribution, int... metaFeatureIndexes) {
        return extract(convert(objects, features, dataset, classDistribution), metaFeatureIndexes);
    }

    public static Instances convert(int objects, int features, double[][] dataset, int[] classDistribution) {
        int sumC = 0;
        for (int v : classDistribution) {
            sumC += v;
        }
        if (sumC != objects) {
            throw new IllegalArgumentException("Sum values of classDistribution should equals to number of objects");
        }

        ArrayList<Attribute> attributes = new ArrayList<>();
        for (int f = 0; f < features; f++) {
            attributes.add(new Attribute("a" + f));
        }

        ArrayList<String> classNames = new ArrayList<>();
        for (int c = 0; c < classDistribution.length; c++) {
            classNames.add("c" + c);
        }
        attributes.add(new Attribute("class", classNames));

        Instances instances = new Instances("data", attributes, objects);
        instances.setClassIndex(features);

        int c = 0;
        int cnt = 0;

        for (int i = 0; i < objects; i++) {
            double[] values = Arrays.copyOf(dataset[i], features + 1);

            while (cnt == classDistribution[c]) {
                ++c;
                cnt = 0;
            }
            values[features] = c;
            ++cnt;

            DenseInstance instance = new DenseInstance(1.0, values);
            instance.setDataset(instances);
            instances.add(instance);
        }

        return instances;
    }

    public static double[] extract(Instances dataset, int... metaFeatureIndexes) {
        metaFeatureIndexes = notEmpty(metaFeatureIndexes);
        int length = metaFeatureIndexes.length;
        double[] metaFeatures = new double[length];

        for (int i = 0; i < length; i++) {
            MetaFeatureExtractor extractor = extractors.get(metaFeatureIndexes[i]);
            try {
                metaFeatures[i] = (extractor.extractValue(dataset));
            } catch (Exception e) {
                throw new IllegalArgumentException(e);
            }
        }

        return metaFeatures;
    }

    public static String[] list() {
        return names.clone();
    }

    public static String name(int id) {
        return names[id];
    }

}
