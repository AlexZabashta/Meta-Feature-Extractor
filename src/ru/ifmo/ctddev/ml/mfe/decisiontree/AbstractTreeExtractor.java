package ru.ifmo.ctddev.ml.mfe.decisiontree;

import java.util.function.ToDoubleFunction;

/**
 * Created by warrior on 21.04.15.
 */
public abstract class AbstractTreeExtractor {

    private final boolean pruneTree;
    private final ToDoubleFunction<WrappedC45DecisionTree> function;

    public AbstractTreeExtractor(boolean pruneTree, ToDoubleFunction<WrappedC45DecisionTree> function) {
        this.pruneTree = pruneTree;
        this.function = function;
    }

    public double extractValue(WrappedC45DecisionTree tree) throws Exception {
        if (pruneTree) {
            return function.applyAsDouble(tree);
        } else {
            return function.applyAsDouble(tree);
        }
    }

    public abstract String getName();

}
