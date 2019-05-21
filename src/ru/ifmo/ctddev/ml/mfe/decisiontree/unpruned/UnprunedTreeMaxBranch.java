package ru.ifmo.ctddev.ml.mfe.decisiontree.unpruned;

import ru.ifmo.ctddev.ml.mfe.decisiontree.TreeMaxBranch;

/**
 * Created by warrior on 23.04.15.
 */
public class UnprunedTreeMaxBranch extends TreeMaxBranch {

    private static final String NAME = "unpruned max branch";
    private static final boolean PRUNE_TREE = false;

    public UnprunedTreeMaxBranch() {
        super(PRUNE_TREE);
    }

    @Override
    public String getName() {
        return NAME;
    }
}
