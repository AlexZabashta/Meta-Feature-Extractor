package ru.ifmo.ctddev.ml.mfe.decisiontree.pruned;

import ru.ifmo.ctddev.ml.mfe.decisiontree.TreeMaxAttr;

/**
 * Created by warrior on 23.04.15.
 */
public class PrunedTreeMaxAttr extends TreeMaxAttr {

    private static final String NAME = "pruned max attr";
    private static final boolean PRUNE_TREE = true;

    public PrunedTreeMaxAttr() {
        super(PRUNE_TREE);
    }

    @Override
    public String getName() {
        return NAME;
    }
}
