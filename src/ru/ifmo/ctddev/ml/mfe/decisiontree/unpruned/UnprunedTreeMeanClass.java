package ru.ifmo.ctddev.ml.mfe.decisiontree.unpruned;

import ru.ifmo.ctddev.ml.mfe.decisiontree.TreeMeanClass;

/**
 * Created by warrior on 23.04.15.
 */
public class UnprunedTreeMeanClass extends TreeMeanClass {

    private static final String NAME = "unpruned mean class";
    private static final boolean PRUNE_TREE = false;

    public UnprunedTreeMeanClass() {
        super(PRUNE_TREE);
    }

    @Override
    public String getName() {
        return NAME;
    }
}
