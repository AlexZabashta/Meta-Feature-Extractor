package ru.ifmo.ctddev.ml.mfe.decisiontree;

/**
 * Created by warrior on 12.05.15.
 */
public abstract class TreeMinAttr extends AbstractTreeExtractor {

    public TreeMinAttr(boolean pruneTree) {
        super(pruneTree, WrappedC45DecisionTree::minAttr);
    }
}
