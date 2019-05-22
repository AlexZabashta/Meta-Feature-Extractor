package ru.ifmo.ctddev.ml.mfe;

import java.util.Random;
import java.util.function.Supplier;
import java.util.function.ToDoubleBiFunction;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class Landmark implements ToDoubleBiFunction<Instances, Random> {

    public final Supplier<Classifier> classifier;

    public final int numFolds;

    public Landmark(Supplier<Classifier> classifier, int numFolds) {
        this.classifier = classifier;
        this.numFolds = numFolds;
    }

    @Override
    public double applyAsDouble(Instances instances, Random random) {
        try {
            Evaluation evaluation = new Evaluation(instances);
            evaluation.crossValidateModel(classifier.get(), instances, numFolds, random);
            double fscore = evaluation.weightedFMeasure();
            if (Double.isFinite(fscore) && 0 <= fscore && fscore <= 1) {
                return fscore;
            } else {
                throw new IllegalStateException("Invalid F-score = " + fscore);
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
            return 0;
        }
    }

    @Override
    public String toString() {
        return numFolds + "-fold CV(" + classifier + ") F-score";
    }

}
