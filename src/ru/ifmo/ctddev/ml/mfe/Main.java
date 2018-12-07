package ru.ifmo.ctddev.ml.mfe;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.StringTokenizer;

import weka.core.DenseInstance;
import weka.core.Instances;

public class Main {

    private static StringTokenizer tokenizer = new StringTokenizer("");

    static String next(BufferedReader reader) throws IOException {
        while (!tokenizer.hasMoreTokens()) {
            tokenizer = new StringTokenizer(reader.readLine());
        }
        return tokenizer.nextToken();
    }

    static int nextInt(BufferedReader reader) throws IOException {
        return Integer.parseInt(next(reader));
    }

    static double nextDouble(BufferedReader reader) throws IOException {
        return Double.parseDouble(next(reader));
    }

    public static void main(String[] args) throws IOException {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(System.in))) {

            int metaFeaturesLength = nextInt(reader);
            int[] metaFeatureIndexes = new int[metaFeaturesLength];

            for (int mfid = 0; mfid < metaFeaturesLength; mfid++) {
                metaFeatureIndexes[mfid] = nextInt(reader);
            }

            while (true) {

                int objects = nextInt(reader), features = nextInt(reader), classes = nextInt(reader);
                Instances instances = Extractor.getInstances(objects, features, classes);

                for (int oid = 0; oid < objects; oid++) {

                    DenseInstance instance = new DenseInstance(features + 1);
                    instance.setDataset(instances);

                    for (int fid = 0; fid <= features; fid++) {
                        instance.setValue(fid, nextDouble(reader));
                    }
                    instances.add(instance);
                }

                double[] metaFeatures = Extractor.extract(instances, metaFeatureIndexes);

                for (int mfid = 0; mfid < metaFeaturesLength; mfid++) {
                    if (mfid != 0) {
                        System.out.print(' ');
                    }
                    System.out.print(metaFeatures[mfid]);
                }
                System.out.flush();               

            }
        }
    }
}
