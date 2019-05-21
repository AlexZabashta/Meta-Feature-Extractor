package ru.ifmo.ctddev.ml.mfe.decisiontree.pruned;

import ru.ifmo.ctddev.ml.mfe.decisiontree.TreeMaxBranch;

/**
 * Created by warrior on 23.04.15.
 */
public class PrunedTreeMaxBranch extends TreeMaxBranch {

    private static final String NAME = "pruned max branch";
    private static final boolean PRUNE_TREE = true;

    public PrunedTreeMaxBranch() {
        super(PRUNE_TREE);
    }

    @Override
    public String getName() {
        return NAME;
    }
}
