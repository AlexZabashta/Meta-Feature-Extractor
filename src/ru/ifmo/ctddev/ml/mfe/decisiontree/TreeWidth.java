package ru.ifmo.ctddev.ml.mfe.decisiontree;

/**
 * Created by warrior on 21.04.15.
 */
public abstract class TreeWidth extends AbstractTreeExtractor {

    public TreeWidth(boolean pruneTree) {
        super(pruneTree, WrappedC45DecisionTree::getWidth);
    }
}
