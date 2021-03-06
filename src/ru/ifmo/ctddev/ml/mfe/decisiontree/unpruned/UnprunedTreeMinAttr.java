package ru.ifmo.ctddev.ml.mfe.decisiontree.unpruned;

import ru.ifmo.ctddev.ml.mfe.decisiontree.TreeMinAttr;

/**
 * Created by warrior on 23.04.15.
 */
public class UnprunedTreeMinAttr extends TreeMinAttr {

    private static final String NAME = "unpruned min attr";
    private static final boolean PRUNE_TREE = false;

    public UnprunedTreeMinAttr() {
        super(PRUNE_TREE);
    }

    @Override
    public String getName() {
        return NAME;
    }
}
