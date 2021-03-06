package ru.ifmo.ctddev.ml.mfe.decisiontree.unpruned;

import ru.ifmo.ctddev.ml.mfe.decisiontree.TreeDevClass;

/**
 * Created by warrior on 23.04.15.
 */
public class UnprunedTreeDevClass extends TreeDevClass {

    private static final String NAME = "unpruned dev class";
    private static final boolean PRUNE_TREE = false;

    public UnprunedTreeDevClass() {
        super(PRUNE_TREE);
    }

    @Override
    public String getName() {
        return NAME;
    }
}
