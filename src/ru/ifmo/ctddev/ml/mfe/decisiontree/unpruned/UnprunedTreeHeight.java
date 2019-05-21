package ru.ifmo.ctddev.ml.mfe.decisiontree.unpruned;

import ru.ifmo.ctddev.ml.mfe.decisiontree.TreeHeight;

/**
 * Created by warrior on 23.04.15.
 */
public class UnprunedTreeHeight extends TreeHeight {

    private static final String NAME = "unpruned height";
    private static final boolean PRUNE_TREE = false;

    public UnprunedTreeHeight() {
        super(PRUNE_TREE);
    }

    @Override
    public String getName() {
        return NAME;
    }
}
