package ru.ifmo.ctddev.ml.mfe.decisiontree.unpruned;

import ru.ifmo.ctddev.ml.mfe.decisiontree.TreeWidth;

/**
 * Created by warrior on 23.04.15.
 */
public class UnprunedTreeWidth extends TreeWidth {

    private static final String NAME = "unpruned width";
    private static final boolean PRUNE_TREE = false;

    public UnprunedTreeWidth() {
        super(PRUNE_TREE);
    }

    @Override
    public String getName() {
        return NAME;
    }
}
