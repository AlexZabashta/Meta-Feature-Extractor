package ru.ifmo.ctddev.ml.mfe;

import java.util.Arrays;

import ru.ifmo.ctddev.ml.mfe.decisiontree.WrappedC45DecisionTree;
import ru.ifmo.ctddev.ml.mfe.decisiontree.WrappedC45ModelSelection;
import ru.ifmo.ctddev.ml.mfe.decisiontree.pruned.PrunedTreeDevAttr;
import ru.ifmo.ctddev.ml.mfe.decisiontree.pruned.PrunedTreeDevBranch;
import ru.ifmo.ctddev.ml.mfe.decisiontree.pruned.PrunedTreeDevClass;
import ru.ifmo.ctddev.ml.mfe.decisiontree.pruned.PrunedTreeDevLevel;
import ru.ifmo.ctddev.ml.mfe.decisiontree.pruned.PrunedTreeHeight;
import ru.ifmo.ctddev.ml.mfe.decisiontree.pruned.PrunedTreeLeavesNumber;
import ru.ifmo.ctddev.ml.mfe.decisiontree.pruned.PrunedTreeMaxAttr;
import ru.ifmo.ctddev.ml.mfe.decisiontree.pruned.PrunedTreeMaxBranch;
import ru.ifmo.ctddev.ml.mfe.decisiontree.pruned.PrunedTreeMaxClass;
import ru.ifmo.ctddev.ml.mfe.decisiontree.pruned.PrunedTreeMaxLevel;
import ru.ifmo.ctddev.ml.mfe.decisiontree.pruned.PrunedTreeMeanAttr;
import ru.ifmo.ctddev.ml.mfe.decisiontree.pruned.PrunedTreeMeanBranch;
import ru.ifmo.ctddev.ml.mfe.decisiontree.pruned.PrunedTreeMeanClass;
import ru.ifmo.ctddev.ml.mfe.decisiontree.pruned.PrunedTreeMeanLevel;
import ru.ifmo.ctddev.ml.mfe.decisiontree.pruned.PrunedTreeMinClass;
import ru.ifmo.ctddev.ml.mfe.decisiontree.pruned.PrunedTreeNodeNumber;
import ru.ifmo.ctddev.ml.mfe.decisiontree.pruned.PrunedTreeWidth;
import weka.classifiers.trees.j48.ModelSelection;
import weka.core.Instances;

public class MetaFeatures {
    public static final int LENGTH = 23;

    static final String[] names = { "meanCorrelation", "meanKurtosis", "meanSkewness", "meanClassVar", "minClassVar", "maxClassVar", "meanInClassDist", "minInClassDist", "maxInClassDist", "meanOutClassDist", "minOutClassDist", "maxOutClassDist", "Tree", "Tree", "Tree", "Tree", "Tree", "Tree", "Tree", "Tree", "Tree",
            "Tree", "Tree", "Tree", "Tree", "Tree", "Tree", "Tree", "Tree" };

    public static String name(int index) {
        return names[index];
    }

    public static double[] extract(int objects, int features, int classes, double[][] data, int[] labels, Instances instances) {
        double[] metaFeatures = new double[LENGTH];
        double[][] tdata = new double[features][objects];

        for (int i = 0; i < objects; i++) {
            for (int j = 0; j < features; j++) {
                tdata[j][i] = data[i][j];
            }
        }

        int mfid = 0;

        mfid = calcCor(mfid, objects, features, tdata, metaFeatures);
        mfid = calcStat(mfid, objects, features, tdata, metaFeatures);
        mfid = calcClassVar(mfid, objects, features, classes, data, labels, metaFeatures);
        mfid = calcClassDist(mfid, objects, features, classes, data, labels, tdata, metaFeatures);
        mfid = calcTreeMF(mfid, instances, metaFeatures);

        return metaFeatures;
    }

    private static int calcCor(int mfid, int objects, int features, double[][] tdata, double[] metaFeatures) {

        double meanCorrelation = 0;

        for (int i = 0; i < features; i++) {
            for (int j = 0; j < i; j++) {
                meanCorrelation += Utils.covariance(tdata[i], tdata[j], 0, 0);
            }
        }
        metaFeatures[mfid++] = meanCorrelation * 2 / (features * (features - 1)); // 0

        return mfid;
    }

    private static int calcStat(int mfid, int objects, int features, double[][] tdata, double[] metaFeatures) {
        double meanKurtosis = 0;
        double meanSkewness = 0;

        for (int fid = 0; fid < features; fid++) {
            double variance = Utils.variance(tdata[fid], 0);
            if (variance > 1e-6) {
                meanKurtosis += Utils.centralMoment(tdata[fid], 4, 0) / Math.pow(variance, 2);
                meanSkewness += Utils.centralMoment(tdata[fid], 3, 0) / Math.pow(variance, 1.5);
            }
        }
        metaFeatures[mfid++] = meanKurtosis / features; // 1
        metaFeatures[mfid++] = meanSkewness / features; // 2
        return mfid;
    }

    private static int calcClassVar(int mfid, int objects, int features, int classes, double[][] data, int[] labels, double[] metaFeatures) {
        double meanClassVar = 0;
        double minClassVar = Double.POSITIVE_INFINITY;
        double maxClassVar = Double.NEGATIVE_INFINITY;

        for (int fid = 0; fid < features; fid++) {
            double classVar = 0;

            double[] sum0 = new double[classes];
            double[] sum1 = new double[classes];
            double[] sum2 = new double[classes];

            for (int oid = 0; oid < objects; oid++) {
                double v = data[oid][fid];
                double p = 1;
                sum0[labels[oid]] += p;
                p *= v;
                sum1[labels[oid]] += p;
                p *= v;
                sum2[labels[oid]] += p;
            }

            for (int label = 0; label < classes; label++) {
                if (sum0[label] > 0) {
                    classVar += (sum2[label] - sum1[label] * sum1[label] / sum0[label]);
                }
            }
            classVar /= objects;

            meanClassVar += classVar;
            minClassVar = Math.min(minClassVar, classVar);
            maxClassVar = Math.max(maxClassVar, classVar);
        }

        metaFeatures[mfid++] = meanClassVar / features; // 3
        metaFeatures[mfid++] = minClassVar; // 4
        metaFeatures[mfid++] = maxClassVar; // 5
        return mfid;
    }

    private static int calcClassDist(int mfid, int objects, int features, int classes, double[][] data, int[] labels, double[][] tdata, double[] metaFeatures) {
        double meanInClassDist = 0;
        double minInClassDist = Double.POSITIVE_INFINITY;
        double maxInClassDist = Double.NEGATIVE_INFINITY;

        double meanOutClassDist = 0;
        double minOutClassDist = Double.POSITIVE_INFINITY;
        double maxOutClassDist = Double.NEGATIVE_INFINITY;

        int[] cnt = new int[classes];
        for (int oid = 0; oid < objects; oid++) {
            ++cnt[labels[oid]];
        }

        double[][] dataPerClass = new double[classes][];
        for (int label = 0; label < classes; label++) {
            dataPerClass[label] = new double[cnt[label]];
        }

        for (int fid = 0; fid < features; fid++) {
            double inClassDist = 0;
            double cntIn = 0;

            Arrays.fill(cnt, 0);

            for (int oid = 0; oid < objects; oid++) {
                int label = labels[oid];
                dataPerClass[label][cnt[label]++] = data[oid][fid];
            }

            for (int label = 0; label < classes; label++) {
                if (dataPerClass[label].length > 1) {
                    inClassDist += dist(dataPerClass[label]);
                    cntIn += dataPerClass[label].length * (dataPerClass[label].length - 1) / 2;
                }
            }

            double allDist = dist(tdata[fid]);
            double outClassDist = allDist - inClassDist;

            if (cntIn > 0) {
                inClassDist /= cntIn;
            }

            double cntOut = objects * (objects - 1) / 2;
            if (cntOut > 0) {
                outClassDist /= cntOut;
            }

            meanInClassDist += inClassDist;
            minInClassDist = Math.min(minInClassDist, inClassDist);
            maxInClassDist = Math.max(maxInClassDist, inClassDist);

            meanOutClassDist += outClassDist;
            minOutClassDist = Math.min(minOutClassDist, outClassDist);
            maxOutClassDist = Math.max(maxOutClassDist, outClassDist);

        }

        metaFeatures[mfid++] = meanInClassDist / features; // 6
        metaFeatures[mfid++] = minInClassDist; // 7
        metaFeatures[mfid++] = maxInClassDist; // 8

        metaFeatures[mfid++] = meanOutClassDist / features; // 9
        metaFeatures[mfid++] = minOutClassDist; // 10
        metaFeatures[mfid++] = maxOutClassDist; // 11
        return mfid;
    }

    private static int calcTreeMF(int mfid, Instances instances, double[] metaFeatures) {

        try {

            ModelSelection modelSelection = new WrappedC45ModelSelection(instances);
            WrappedC45DecisionTree tree = new WrappedC45DecisionTree(modelSelection, true);
            tree.buildClassifier(instances);

            metaFeatures[mfid++] = (new PrunedTreeDevAttr()).extractValue(tree); // 12
            metaFeatures[mfid++] = (new PrunedTreeDevBranch()).extractValue(tree); // 13
            metaFeatures[mfid++] = (new PrunedTreeDevLevel()).extractValue(tree); // 14
            metaFeatures[mfid++] = (new PrunedTreeHeight()).extractValue(tree); // 15
            metaFeatures[mfid++] = (new PrunedTreeMaxAttr()).extractValue(tree); // 16
            metaFeatures[mfid++] = (new PrunedTreeMaxLevel()).extractValue(tree); // 17
            metaFeatures[mfid++] = (new PrunedTreeMeanBranch()).extractValue(tree); // 18
            metaFeatures[mfid++] = (new PrunedTreeMeanLevel()).extractValue(tree); // 19
            metaFeatures[mfid++] = (new PrunedTreeNodeNumber()).extractValue(tree); // 20
            metaFeatures[mfid++] = (new PrunedTreeWidth()).extractValue(tree); // 21
            metaFeatures[mfid++] = (new PrunedTreeDevClass()).extractValue(tree); // 22
            // metaFeatures[mfid++] = (new PrunedTreeMaxClass()).extractValue(tree); // 23
            // metaFeatures[mfid++] = (new PrunedTreeMinClass()).extractValue(tree); // 24
            // metaFeatures[mfid++] = (new PrunedTreeMeanClass()).extractValue(tree); // 25
            // metaFeatures[mfid++] = (new PrunedTreeLeavesNumber()).extractValue(tree); // 16
            // metaFeatures[mfid++] = (new PrunedTreeMaxBranch()).extractValue(tree); // 17
            // metaFeatures[mfid++] = (new PrunedTreeMeanAttr()).extractValue(tree); // 19
        } catch (Exception e) {
            e.printStackTrace();
        }

        return mfid;
    }

    private static double dist(double[] x) {
        Arrays.sort(x);

        double dist = 0;

        double sum = 0;
        double cnt = 0;

        for (double v : x) {
            dist += v * cnt - sum;
            sum += v;
            cnt += 1;
        }

        return 2 * dist;
    }

}
