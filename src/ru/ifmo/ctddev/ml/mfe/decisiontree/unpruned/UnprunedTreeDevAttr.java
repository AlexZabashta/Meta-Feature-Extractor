package ru.ifmo.ctddev.ml.mfe.decisiontree.unpruned;

import ru.ifmo.ctddev.ml.mfe.decisiontree.TreeDevAttr;

/**
 * Created by warrior on 23.04.15.
 */
public class UnprunedTreeDevAttr extends TreeDevAttr {

    private static final String NAME = "unpruned dev attr";
    private static final boolean PRUNE_TREE = false;

    public UnprunedTreeDevAttr() {
        super(PRUNE_TREE);
    }

    @Override
    public String getName() {
        return NAME;
    }
}
