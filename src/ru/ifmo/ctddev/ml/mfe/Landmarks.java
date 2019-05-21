package ru.ifmo.ctddev.ml.mfe;

import java.util.Random;
import java.util.function.Supplier;

import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class Landmarks {
    @SuppressWarnings("unchecked")
    private static Supplier<Classifier>[] classifiers = new Supplier[] { new Supplier<Classifier>() {

        @Override
        public Classifier get() {
            return new IBk(3);
        }

        @Override
        public String toString() {
            return "kNN";
        }
    }, new Supplier<Classifier>() {

        @Override
        public Classifier get() {
            SMO smo = new SMO();
            smo.setRandomSeed(DEFAULT_SEED);
            return smo;
        }

        @Override
        public String toString() {
            return "SVM";
        }
    }, new Supplier<Classifier>() {

        @Override
        public Classifier get() {
            J48 dt = new J48();
            dt.setSeed(DEFAULT_SEED);
            return dt;
        }

        @Override
        public String toString() {
            return "DT";
        }
    } };

    public static final int DEFAULT_SEED = 42;
    public static final int NUM_CV_FOLDS = 8;
    public static final int NUM_REPEATS = 4;
    public static final int LENGTH = classifiers.length;

    private static Landmark[] landmarks = new Landmark[LENGTH];
    static {
        for (int i = 0; i < LENGTH; i++) {
            landmarks[i] = new Landmark(classifiers[i], NUM_CV_FOLDS);
        }
    }

    public static double[] extract(Instances instances) {
        double[] scores = new double[LENGTH];

        for (int i = 0; i < LENGTH; i++) {
            for (int rep = 0; rep < NUM_REPEATS; rep++) {
                scores[i] += landmarks[i].applyAsDouble(instances, new Random(DEFAULT_SEED + rep));
            }
            scores[i] /= NUM_REPEATS;
        }

        return scores;
    }

    public static String name(int index) {
        return NUM_REPEATS + " x " + landmarks[index];
    }

}
