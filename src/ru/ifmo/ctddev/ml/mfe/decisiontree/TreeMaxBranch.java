package ru.ifmo.ctddev.ml.mfe.decisiontree;

/**
 * Created by warrior on 23.04.15.
 */
public abstract class TreeMaxBranch extends AbstractTreeExtractor {

    public TreeMaxBranch(boolean pruneTree) {
        super(pruneTree, WrappedC45DecisionTree::maxBranch);
    }
}
