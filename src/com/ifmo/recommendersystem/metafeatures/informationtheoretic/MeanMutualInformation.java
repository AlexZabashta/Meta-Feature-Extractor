package com.ifmo.recommendersystem.metafeatures.informationtheoretic;

import com.ifmo.recommendersystem.metafeatures.MetaFeatureExtractor;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.core.Instances;

/**
 * Created by warrior on 23.03.15.
 */
public class MeanMutualInformation extends MetaFeatureExtractor {

    public static final String NAME = "Mean mutual information of class and attribute";

    @Override
    public String getName() {
        return NAME;
    }

    @Override
    public double extractValue(Instances instances) throws Exception {

        try {
            InfoGainAttributeEval infoGain = new InfoGainAttributeEval();

            infoGain.buildEvaluator(instances);

            double meanMutualInformation = 0;
            int norm = 0;
            for (int i = 0; i < instances.numAttributes(); i++) {
                if (i != instances.classIndex()) {

                    double mutualInformation = infoGain.evaluateAttribute(i);

                    if (Double.isNaN(mutualInformation) || Double.isInfinite(mutualInformation)) {
                        continue;
                    }

                    meanMutualInformation += mutualInformation;
                    norm += 1;

                }
            }

            if (norm == 0) {
                return 0;
            } else {
                return meanMutualInformation / norm;
            }
        } catch (Exception exception) {
            return 0;
        }

    }
}
