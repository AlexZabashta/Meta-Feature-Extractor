package ru.ifmo.ctddev.ml.mfe;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Locale;
import java.util.Scanner;

import weka.core.Instances;

public class JointDecMF {
    public static double[][] sqrMF = new double[MetaFeatures.LENGTH][MetaFeatures.LENGTH];
    public static double[][] sqrLM = new double[Landmarks.LENGTH][Landmarks.LENGTH];

    public static double[] meanMF = new double[MetaFeatures.LENGTH];
    public static double[] meanLM = new double[Landmarks.LENGTH];

    static {
        try (Scanner scanner = new Scanner(new File("meta" + "_mean.txt"))) {
            scanner.useLocale(Locale.ENGLISH);
            for (int i = 0; i < meanMF.length; i++) {
                meanMF[i] = scanner.nextDouble();
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        try (Scanner scanner = new Scanner(new File("land" + "_mean.txt"))) {
            scanner.useLocale(Locale.ENGLISH);
            for (int i = 0; i < meanLM.length; i++) {
                meanLM[i] = scanner.nextDouble();
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        try (Scanner scanner = new Scanner(new File("meta" + "_sqr.txt"))) {
            scanner.useLocale(Locale.ENGLISH);
            for (int i = 0; i < sqrMF.length; i++) {
                for (int j = 0; j < sqrMF.length; j++) {
                    sqrMF[i][j] = scanner.nextDouble();
                }
            }
        } catch (FileNotFoundException e) {
            for (int i = 0; i < sqrMF.length; i++) {
                sqrMF[i][i] = 1.0;
            }
            e.printStackTrace();
        }
        try (Scanner scanner = new Scanner(new File("land" + "_sqr.txt"))) {
            scanner.useLocale(Locale.ENGLISH);
            for (int i = 0; i < sqrLM.length; i++) {
                for (int j = 0; j < sqrLM.length; j++) {
                    sqrLM[i][j] = scanner.nextDouble();
                }
            }
        } catch (FileNotFoundException e) {
            for (int i = 0; i < sqrLM.length; i++) {
                sqrLM[i][i] = 1.0;
            }
            e.printStackTrace();
        }
    }
    public static final int LENGTH = MetaFeatures.LENGTH + Landmarks.LENGTH;

    public static double[] extract(int objects, int features, int classes, double[][] data, int[] labels) {

        Utils.normalize(objects, features, data);
        Instances instances = Utils.convert(objects, features, classes, data, labels);

        double[] vectMF = MetaFeatures.extract(objects, features, classes, data, labels, instances);
        double[] vectLM = Landmarks.extract(instances);
        double[] vect = new double[LENGTH];

        for (int i = 0; i < vectMF.length; i++) {
            for (int j = 0; j < vectMF.length; j++) {
                vect[i] += (vectMF[j] - meanMF[j]) * sqrMF[j][i];
            }
        }
        for (int i = 0; i < vectLM.length; i++) {
            for (int j = 0; j < vectLM.length; j++) {
                vect[i + vectMF.length] += (vectLM[j] - meanLM[j]) * sqrLM[j][i];
            }
        }

        return vect;
    }

}
