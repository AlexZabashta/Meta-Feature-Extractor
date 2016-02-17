# Recommender system

* result of recommender system training: https://yadi.sk/d/PvwkEynop2eC9
* sample: https://github.com/Undin/recommender-system/blob/master/src/main/java/com/ifmo/recommendersystem/Main.java

P.S. there aren't some meta-features for `kdd_ipums_la_97-small`, `kdd_ipums_la_98-small`, `kdd_ipums_la_99-small`, `mushroom`, 
`pendigits`, `splice`, `sylva_agnostic`, `sylva_prior` (knn and neural meta-features).

# Files
* `algorithms.json` - list of used feature subset selection algorithms. Contains short name, full class names and options
* `classifiers.json` - list of used classifier algorithms. Contains short name, full class names and options
* `config.json` - config for recommender system builder
* evaluation configs (such as `evaluationConfig.json`, `fastEvaluationConfig.json` and etc.) - configs with parameters to use recommender system

# How to use
* download results and unzip to root of project
* change `DATASETS` to array with paths to your datasets
* run `Main`
* ...
* PROFIT!!! - recommendation will be located in `OUTPUT_DIRECTORY`

# todo
create console util
