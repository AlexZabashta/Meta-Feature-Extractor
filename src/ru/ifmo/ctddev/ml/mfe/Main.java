package ru.ifmo.ctddev.ml.mfe;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.StringTokenizer;

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
        double value = Double.parseDouble(next(reader));
        if (Double.isFinite(value)) {
            return value;
        } else {
            throw new IllegalArgumentException("Wrong value " + value);
        }
    }

    public static void main(String[] args) throws IOException {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(System.in))) {

            while (true) {

                int objects = nextInt(reader), features = nextInt(reader), classes = nextInt(reader);

                double[][] data = new double[objects][features];
                int[] labels = new int[objects];

                for (int oid = 0; oid < objects; oid++) {
                    for (int fid = 0; fid < features; fid++) {
                        data[oid][fid] = nextDouble(reader);
                    }
                    labels[oid] = nextInt(reader);

                    if (labels[oid] < 0 || classes <= labels[oid]) {
                        throw new IllegalArgumentException("Wrong label " + labels[oid]);
                    }

                }

                Utils.normalize(objects, features, data);
                Instances instances = Utils.convert(objects, features, classes, data, labels);

                double[] metaFeatures = MetaFeatures.extract(objects, features, classes, data, labels, instances);

                for (int mfid = 0; mfid < metaFeatures.length; mfid++) {
                    if (mfid != 0) {
                        System.out.print(' ');
                    }
                    System.out.print(metaFeatures[mfid]);
                }
                System.out.println();
                System.out.flush();

            }
        }
    }
}
