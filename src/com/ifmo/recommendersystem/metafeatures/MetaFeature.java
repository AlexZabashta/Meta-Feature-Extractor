package com.ifmo.recommendersystem.metafeatures;

/**
 * Created by warrior on 11.04.15.
 */
public class MetaFeature {

	private final String extractorClassName;
	private final double value;

	public MetaFeature(MetaFeatureExtractor extractor, double value) {
		this.extractorClassName = extractor.getClass().getCanonicalName();
		this.value = value;
	}

	public MetaFeature(String extractorClassName, double value) {
		this.extractorClassName = extractorClassName;
		this.value = value;
	}

	public String getExtractorClassName() {
		return extractorClassName;
	}

	public double getValue() {
		return value;
	}

}
