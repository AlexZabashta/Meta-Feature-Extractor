package ru.ifmo.ctddev.ml.mfe.decisiontree;

/**
 * Created by warrior on 12.05.15.
 */
public abstract class TreeDevAttr extends AbstractTreeExtractor {

    public TreeDevAttr(boolean pruneTree) {
        super(pruneTree, WrappedC45DecisionTree::devAttr);
    }
}
