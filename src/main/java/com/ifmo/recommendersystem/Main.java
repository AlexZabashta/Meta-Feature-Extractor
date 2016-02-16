package com.ifmo.recommendersystem;

import com.ifmo.recommendersystem.config.EvaluationConfig;
import com.ifmo.recommendersystem.metafeatures.MetaFeatureConverter;
import com.ifmo.recommendersystem.utils.InstancesUtils;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.json.JSONObject;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by warrior on 09.02.16.
 */
public class Main {

    private static final String EVALUATION_CONFIG = "fastEvaluationConfig.json";
    private static final String OUTPUT_DIRECTORY = "recommendation-results";

    // 
    private static final String[] DATASETS = {
            "datasets-extra/FisherYatesShuffle_FastMiner.arff",
            "datasets-extra/FisherYatesShuffle_SimpleMiner.arff",
            "datasets-extra/GaussGenerator_FastMiner.arff",
            "datasets-extra/GaussGenerator_SimpleMiner.arff",
            "datasets-extra/SeveralSwapsGenerator_FastMiner.arff",
            "datasets-extra/SeveralSwapsGenerator_SimpleMiner.arff"
    };

    public static void main(String[] args) throws Exception {
        // create config
        EvaluationConfig config = new EvaluationConfig(EVALUATION_CONFIG);
        // create earr matrices
        List<double[][]> matrices = config.getClassifiers().stream()
                .map(config::createEarrMatrix)
                .collect(Collectors.toList());
        // read preprocess meta features for some base datasets
        Instances metaFeaturesList = MetaFeatureConverter.createInstances(config, RecommenderSystemBuilder.META_FEATURES_DIRECTORY, "metaFeatures");

        File resultDir = new File(OUTPUT_DIRECTORY);
        if (!resultDir.exists()) {
            resultDir.mkdirs();
        }

        for (String datasetFilename : DATASETS) {
            System.out.println(datasetFilename);

            // read dataset
            Instances dataset = InstancesUtils.createInstances(datasetFilename);

            // calculate meta features for given dataset
            double[] values = config.getExtractors().stream()
                    .parallel()
                    .mapToDouble(e -> {
                        System.out.println(e.getName());
                        return e.extract(dataset).getValue();
                    })
                    .toArray();
            System.out.println();
            
            Instance metaFeatures = new DenseInstance(1, values);

            File resultFile = new File(resultDir, FilenameUtils.getBaseName(datasetFilename) + ".json");
            JSONObject result = new JSONObject();

            for (int i = 0; i < matrices.size(); i++) {
                double[][] matrix = matrices.get(i);
                RecommenderSystem recommenderSystem = new RecommenderSystem(matrix, metaFeaturesList, config.getAlgorithms());
                List<FSSAlgorithm> algorithms = recommenderSystem.recommend(metaFeatures);

                // do something with result. For example print algorithms names
                List<String> names = algorithms.stream()
                        .map(FSSAlgorithm::getName)
                        .collect(Collectors.toList());

                String classifierName = config.getClassifiers().get(i).getName();
                result.put(classifierName, names);
            }

            FileUtils.write(resultFile, result.toString(4));
        }
    }
}
