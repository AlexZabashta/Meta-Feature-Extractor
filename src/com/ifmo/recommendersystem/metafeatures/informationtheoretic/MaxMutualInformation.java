package com.ifmo.recommendersystem.metafeatures.informationtheoretic;

import com.ifmo.recommendersystem.metafeatures.MetaFeatureExtractor;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.core.Instances;

/**
 * Created by warrior on 23.03.15.
 */
public class MaxMutualInformation extends MetaFeatureExtractor {

    public static final String NAME = "Maximum mutual information";

    @Override
    public String getName() {
        return NAME;
    }

    @Override
    public double extractValue(Instances instances) throws Exception {
        try {
            InfoGainAttributeEval infoGain = new InfoGainAttributeEval();

            infoGain.buildEvaluator(instances);

            double maxMutualInformation = 0;
            for (int i = 0; i < instances.numAttributes(); i++) {
                if (i != instances.classIndex()) {

                    double mutualInformation = infoGain.evaluateAttribute(i);

                    if (Double.isNaN(mutualInformation) || Double.isInfinite(mutualInformation)) {
                        continue;
                    }

                    maxMutualInformation = Math.max(maxMutualInformation, mutualInformation);
                }
            }
            return maxMutualInformation;
        } catch (Exception exception) {
            return 0;
        }
    }
}
