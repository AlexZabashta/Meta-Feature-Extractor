package ru.ifmo.ctddev.ml.mfe.decisiontree.pruned;

import ru.ifmo.ctddev.ml.mfe.decisiontree.TreeWidth;

/**
 * Created by warrior on 23.04.15.
 */
public class PrunedTreeWidth extends TreeWidth {

    private static final String NAME = "pruned width";
    private static final boolean PRUNE_TREE = true;

    public PrunedTreeWidth() {
        super(PRUNE_TREE);
    }

    @Override
    public String getName() {
        return NAME;
    }
}
