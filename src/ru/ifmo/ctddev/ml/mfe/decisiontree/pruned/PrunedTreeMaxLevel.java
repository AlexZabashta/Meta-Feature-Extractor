package ru.ifmo.ctddev.ml.mfe.decisiontree.pruned;

import ru.ifmo.ctddev.ml.mfe.decisiontree.TreeMaxLevel;

/**
 * Created by warrior on 23.04.15.
 */
public class PrunedTreeMaxLevel extends TreeMaxLevel {

    private static final String NAME = "pruned max level";
    private static final boolean PRUNE_TREE = true;

    public PrunedTreeMaxLevel() {
        super(PRUNE_TREE);
    }

    @Override
    public String getName() {
        return NAME;
    }
}
