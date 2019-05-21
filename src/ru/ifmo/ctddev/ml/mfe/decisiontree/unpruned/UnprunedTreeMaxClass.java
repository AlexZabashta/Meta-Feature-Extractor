package ru.ifmo.ctddev.ml.mfe.decisiontree.unpruned;

import ru.ifmo.ctddev.ml.mfe.decisiontree.TreeMaxClass;

/**
 * Created by warrior on 23.04.15.
 */
public class UnprunedTreeMaxClass extends TreeMaxClass {

    private static final String NAME = "unpruned max class";
    private static final boolean PRUNE_TREE = false;

    public UnprunedTreeMaxClass() {
        super(PRUNE_TREE);
    }

    @Override
    public String getName() {
        return NAME;
    }
}
