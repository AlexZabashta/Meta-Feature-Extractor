package com.ifmo.recommendersystem.metafeatures;

import weka.core.Attribute;
import weka.core.Instances;

import java.util.Arrays;

public abstract class MetaFeatureExtractor {
    public abstract String getName();

    public abstract double extractValue(Instances instances) throws Exception;

    protected boolean isNonClassAttributeWithType(Instances instances, int attributeIndex, int... types) {
        Attribute attribute = instances.attribute(attributeIndex);
        int type = attribute.type();
        return attributeIndex != instances.classIndex() && Arrays.stream(types).anyMatch(t -> t == type);
    }

    /**
     * Creates a new instance of a meta feature extractor given it's class name.
     *
     * @param className
     *            the fully qualified class name of the meta feature extractor
     * @return new instance of the meta feature extractor. If the extractor name is invalid return null
     * @throws ClassNotFoundException
     * @throws IllegalAccessException
     * @throws InstantiationException
     */
    public static MetaFeatureExtractor forName(String className) throws ClassNotFoundException, InstantiationException, IllegalAccessException {
        Class<?> clazz;

        clazz = Class.forName(className);

        if (!MetaFeatureExtractor.class.isAssignableFrom(clazz)) {
            return null;
        }
        Object newInstance;

        newInstance = clazz.newInstance();

        return (MetaFeatureExtractor) newInstance;
    }
}
