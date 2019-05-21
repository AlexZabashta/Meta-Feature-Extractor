package ru.ifmo.ctddev.ml.mfe;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Objects;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

public class Utils {

    public static double centralMoment(double[] values, int k) {
        return centralMoment(values, k, mean(values));
    }

    public static double centralMoment(double[] values, int k, double mean) {
        return rawMoment(unshifted(values, mean), k);
    }

    private static void checkSameLength(double[] values1, double[] values2) {
        values1 = Objects.requireNonNull(values1);
        values2 = Objects.requireNonNull(values2);
        if (values1.length != values2.length) {
            throw new IllegalArgumentException("value arrays must have same length");
        }
    }

    public static Instances convert(int objects, int features, int classes, double[][] data, int[] labels) {
        Instances instances = getInstances(objects, features, classes);

        for (int oid = 0; oid < objects; oid++) {
            double[] values = Arrays.copyOf(data[oid], features + 1);
            values[features] = labels[oid];

            DenseInstance instance = new DenseInstance(1.0, values);
            instance.setDataset(instances);
            instances.add(instance);
        }

        return instances;
    }

    public static double covariance(double[] values1, double[] values2) {
        checkSameLength(values1, values2);
        return covariance(values1, values2, mean(values1), mean(values2));
    }

    public static double covariance(double[] values1, double[] values2, double mean1, double mean2) {
        checkSameLength(values1, values2);
        int length = values1.length;
        double cov = 0;
        double count = 0;
        for (int i = 0; i < length; i++) {
            if (isCorrectValue(values1[i]) && isCorrectValue(values2[i])) {
                cov += (values1[i] - mean1) * (values2[i] - mean2);
                count++;
            }
        }
        cov /= count;
        return cov;
    }

    public static Instances getInstances(int objects, int features, int classes) {
        ArrayList<Attribute> attributes = new ArrayList<>();
        for (int f = 0; f < features; f++) {
            attributes.add(new Attribute("a" + f));
        }

        ArrayList<String> classNames = new ArrayList<>();
        for (int c = 0; c < classes; c++) {
            classNames.add("c" + c);
        }
        attributes.add(new Attribute("class", classNames));

        Instances instances = new Instances("data", attributes, objects);
        instances.setClassIndex(features);
        return instances;

    }

    public static boolean isCorrectValue(double v) {
        return Double.isFinite(v);
    }

    public static double linearCorrelationCoefficient(double covariance, double variance1, double variance2) {
        double norm = Math.sqrt(variance1 * variance2);
        if (norm > 1e-9) {
            return covariance / norm;
        } else {
            return 0;
        }
    }

    public static double linearCorrelationCoefficient(double[] values1, double[] values2) {
        checkSameLength(values1, values2);
        double mean1 = mean(values1);
        double mean2 = mean(values2);
        double variance1 = variance(values1, mean1);
        double variance2 = variance(values2, mean2);
        return linearCorrelationCoefficient(values1, values2, mean1, mean2, variance1, variance2);
    }

    public static double linearCorrelationCoefficient(double[] values1, double[] values2, double mean1, double mean2, double variance1, double variance2) {
        checkSameLength(values1, values2);
        double covariance = covariance(values1, values2, mean1, mean2);
        return linearCorrelationCoefficient(covariance, variance1, variance2);
    }

    public static double mean(double[] values) {
        return rawMoment(values, 1);
    }

    public static void normalize(int objects, int features, double[][] data) {
        for (int fid = 0; fid < features; fid++) {
            double mean = 0;
            for (int oid = 0; oid < objects; oid++) {
                mean += data[oid][fid];
            }
            mean /= objects;

            double var = 0, skw = 0;
            for (int oid = 0; oid < objects; oid++) {
                data[oid][fid] -= mean;
                double s = data[oid][fid] * data[oid][fid];
                var += s;
                skw += s * data[oid][fid];
            }
            var /= objects;

            double sigma = 0;

            if (var > 1e-6) {
                sigma = 1 / Math.sqrt(var);
                if (skw < 0) {
                    sigma *= -1;
                }
            }
            for (int oid = 0; oid < objects; oid++) {
                data[oid][fid] *= sigma;
            }
        }
    }

    public static double rawMoment(double[] values, int k) {
        double acc = 0;
        double count = 0;
        for (double value : values) {
            if (isCorrectValue(value)) {
                acc += StrictMath.pow(value, k);
                count++;
            }
        }
        return acc / count;
    }

    private static double[] unshifted(double[] values, double mean) {
        double[] tmp = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            if (isCorrectValue(values[i])) {
                tmp[i] = values[i] - mean;
            }
        }
        return tmp;
    }

    public static double variance(double[] values) {
        return centralMoment(values, 2);
    }

    public static double variance(double[] values, double mean) {
        return centralMoment(values, 2, mean);
    }

}
