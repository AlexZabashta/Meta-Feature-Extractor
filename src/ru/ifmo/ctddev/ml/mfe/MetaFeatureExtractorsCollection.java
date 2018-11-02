package ru.ifmo.ctddev.ml.mfe;

import java.util.ArrayList;
import java.util.List;

import com.ifmo.recommendersystem.metafeatures.MetaFeatureExtractor;

public class MetaFeatureExtractorsCollection {

    public static void main(String[] args) {
        List<MetaFeatureExtractor> extractors = rat();

        for (int i = 0; i < 26; i++) {
            System.out.println(i + " " + extractors.get(i).getName());
        }

    }

    public static List<MetaFeatureExtractor> tree4() {
        List<MetaFeatureExtractor> e = new ArrayList<MetaFeatureExtractor>();
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeMeanAttr());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeMeanLevel());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeMeanBranch());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeMeanClass());
        return e;
    }

    public static List<MetaFeatureExtractor> treestat4() {
        List<MetaFeatureExtractor> e = new ArrayList<MetaFeatureExtractor>();

        e.add(new com.ifmo.recommendersystem.metafeatures.statistical.MeanLinearCorrelationCoefficient());
        e.add(new com.ifmo.recommendersystem.metafeatures.informationtheoretic.MeanMutualInformation());

        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeMeanBranch());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeMeanClass());
        return e;
    }

    public static List<MetaFeatureExtractor> top3diff() {
        List<MetaFeatureExtractor> e = new ArrayList<MetaFeatureExtractor>();

        e.add(new com.ifmo.recommendersystem.metafeatures.statistical.MeanLinearCorrelationCoefficient());
        e.add(new com.ifmo.recommendersystem.metafeatures.informationtheoretic.MeanMutualInformation());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeMeanBranch());
        return e;
    }

    public static List<MetaFeatureExtractor> rat() {
        List<MetaFeatureExtractor> e = new ArrayList<MetaFeatureExtractor>();

        e.add(new com.ifmo.recommendersystem.metafeatures.statistical.MeanLinearCorrelationCoefficient());

        e.add(new com.ifmo.recommendersystem.metafeatures.statistical.MeanSkewness());
        e.add(new com.ifmo.recommendersystem.metafeatures.statistical.MeanKurtosis());

        e.add(new com.ifmo.recommendersystem.metafeatures.informationtheoretic.NormalizedClassEntropy());
        e.add(new com.ifmo.recommendersystem.metafeatures.informationtheoretic.MeanNormalizedFeatureEntropy());
        e.add(new com.ifmo.recommendersystem.metafeatures.informationtheoretic.MeanMutualInformation());
        e.add(new com.ifmo.recommendersystem.metafeatures.informationtheoretic.MaxMutualInformation());
        e.add(new com.ifmo.recommendersystem.metafeatures.informationtheoretic.EquivalentNumberOfFeatures());
        e.add(new com.ifmo.recommendersystem.metafeatures.informationtheoretic.NoiseSignalRatio());

        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeDevAttr());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeDevBranch());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeDevLevel());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeHeight());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeLeavesNumber());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeMaxAttr());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeMaxBranch());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeMaxLevel());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeMeanAttr());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeMeanBranch());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeMeanLevel());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeNodeNumber());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeWidth());

        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeDevClass());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeMaxClass());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeMinClass());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeMeanClass());
        return e;

    }

    public static List<MetaFeatureExtractor> all() {
        List<MetaFeatureExtractor> e = new ArrayList<MetaFeatureExtractor>();

        e.add(new com.ifmo.recommendersystem.metafeatures.general.NumberOfInstances());
        e.add(new com.ifmo.recommendersystem.metafeatures.general.NumberOfFeatures());
        e.add(new com.ifmo.recommendersystem.metafeatures.general.NumberOfClasses());
        e.add(new com.ifmo.recommendersystem.metafeatures.general.DataSetDimensionality());
        e.add(new com.ifmo.recommendersystem.metafeatures.statistical.MeanLinearCorrelationCoefficient());
        e.add(new com.ifmo.recommendersystem.metafeatures.statistical.MeanSkewness());
        e.add(new com.ifmo.recommendersystem.metafeatures.statistical.MeanKurtosis());
        e.add(new com.ifmo.recommendersystem.metafeatures.informationtheoretic.NormalizedClassEntropy());
        e.add(new com.ifmo.recommendersystem.metafeatures.informationtheoretic.MeanNormalizedFeatureEntropy());
        e.add(new com.ifmo.recommendersystem.metafeatures.informationtheoretic.MeanMutualInformation());
        e.add(new com.ifmo.recommendersystem.metafeatures.informationtheoretic.MaxMutualInformation());
        e.add(new com.ifmo.recommendersystem.metafeatures.informationtheoretic.EquivalentNumberOfFeatures());
        e.add(new com.ifmo.recommendersystem.metafeatures.informationtheoretic.NoiseSignalRatio());

        e.add(new com.ifmo.recommendersystem.metafeatures.statistical.MeanStandardDeviation());
        e.add(new com.ifmo.recommendersystem.metafeatures.statistical.MeanCoefficientOfVariation());

        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeDevAttr());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeDevBranch());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeDevLevel());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeHeight());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeLeavesNumber());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeMaxAttr());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeMaxBranch());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeMaxLevel());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeMeanAttr());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeMeanBranch());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeMeanLevel());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeMinAttr());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeMinBranch());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeNodeNumber());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeWidth());

        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeDevClass());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeMaxClass());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeMinClass());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.pruned.PrunedTreeMeanClass());

        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.unpruned.UnprunedTreeDevAttr());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.unpruned.UnprunedTreeDevBranch());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.unpruned.UnprunedTreeDevLevel());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.unpruned.UnprunedTreeHeight());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.unpruned.UnprunedTreeLeavesNumber());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.unpruned.UnprunedTreeMaxAttr());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.unpruned.UnprunedTreeMaxBranch());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.unpruned.UnprunedTreeMaxLevel());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.unpruned.UnprunedTreeMeanAttr());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.unpruned.UnprunedTreeMeanBranch());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.unpruned.UnprunedTreeMeanLevel());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.unpruned.UnprunedTreeMinAttr());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.unpruned.UnprunedTreeMinBranch());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.unpruned.UnprunedTreeNodeNumber());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.unpruned.UnprunedTreeWidth());

        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.unpruned.UnprunedTreeDevClass());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.unpruned.UnprunedTreeMaxClass());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.unpruned.UnprunedTreeMinClass());
        e.add(new com.ifmo.recommendersystem.metafeatures.decisiontree.unpruned.UnprunedTreeMeanClass());

        e.add(new com.ifmo.recommendersystem.metafeatures.classifierbased.knn.MaxHalfBestK());
        e.add(new com.ifmo.recommendersystem.metafeatures.classifierbased.knn.MaxSqrtBestK());
        e.add(new com.ifmo.recommendersystem.metafeatures.classifierbased.knn.MaxOneTenthBestK());
        e.add(new com.ifmo.recommendersystem.metafeatures.classifierbased.knn.MinHalfBestK());
        e.add(new com.ifmo.recommendersystem.metafeatures.classifierbased.knn.MinSqrtBestK());
        e.add(new com.ifmo.recommendersystem.metafeatures.classifierbased.knn.MinOneTenthBestK());
        e.add(new com.ifmo.recommendersystem.metafeatures.classifierbased.knn.MeanHalfBestK());
        e.add(new com.ifmo.recommendersystem.metafeatures.classifierbased.knn.MeanSqrtBestK());
        e.add(new com.ifmo.recommendersystem.metafeatures.classifierbased.knn.MeanOneTenthBestK());
        e.add(new com.ifmo.recommendersystem.metafeatures.classifierbased.knn.StdDevHalfBestK());
        e.add(new com.ifmo.recommendersystem.metafeatures.classifierbased.knn.StdDevSqrtBestK());
        e.add(new com.ifmo.recommendersystem.metafeatures.classifierbased.knn.StdDevOneTenthBestK());
        e.add(new com.ifmo.recommendersystem.metafeatures.classifierbased.knn.FullBestK());

        e.add(new com.ifmo.recommendersystem.metafeatures.classifierbased.neural.MaxHalfPerceptronWeightSum());
        e.add(new com.ifmo.recommendersystem.metafeatures.classifierbased.neural.MaxSqrtPerceptronWeightSum());
        e.add(new com.ifmo.recommendersystem.metafeatures.classifierbased.neural.MaxOneTenthPerceptronWeightSum());
        e.add(new com.ifmo.recommendersystem.metafeatures.classifierbased.neural.MinHalfPerceptronWeightSum());
        e.add(new com.ifmo.recommendersystem.metafeatures.classifierbased.neural.MinSqrtPerceptronWeightSum());
        e.add(new com.ifmo.recommendersystem.metafeatures.classifierbased.neural.MinOneTenthPerceptronWeightSum());
        e.add(new com.ifmo.recommendersystem.metafeatures.classifierbased.neural.MeanHalfPerceptronWeightSum());
        e.add(new com.ifmo.recommendersystem.metafeatures.classifierbased.neural.MeanSqrtPerceptronWeightSum());
        e.add(new com.ifmo.recommendersystem.metafeatures.classifierbased.neural.MeanOneTenthPerceptronWeightSum());
        e.add(new com.ifmo.recommendersystem.metafeatures.classifierbased.neural.StdDevHalfPerceptronWeightSum());
        e.add(new com.ifmo.recommendersystem.metafeatures.classifierbased.neural.StdDevSqrtPerceptronWeightSum());
        e.add(new com.ifmo.recommendersystem.metafeatures.classifierbased.neural.StdDevOneTenthPerceptronWeightSum());
        e.add(new com.ifmo.recommendersystem.metafeatures.classifierbased.neural.FullPerceptronWeightSum());

        return e;
    }
}
