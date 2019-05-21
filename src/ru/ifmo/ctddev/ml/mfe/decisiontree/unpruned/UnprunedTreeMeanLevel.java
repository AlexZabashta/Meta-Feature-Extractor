package ru.ifmo.ctddev.ml.mfe.decisiontree.unpruned;

import ru.ifmo.ctddev.ml.mfe.decisiontree.TreeMeanLevel;

/**
 * Created by warrior on 23.04.15.
 */
public class UnprunedTreeMeanLevel extends TreeMeanLevel {

    private static final String NAME = "unpruned mean level";
    private static final boolean PRUNE_TREE = false;

    public UnprunedTreeMeanLevel() {
        super(PRUNE_TREE);
    }

    @Override
    public String getName() {
        return NAME;
    }
}
