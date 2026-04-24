from jax_gemini.prompts.intent import IntentClassifier


class TestIntentClassifier:
    def test_classify_build(self):
        assert IntentClassifier.classify("build a model") == "build"
        assert IntentClassifier.classify("create an mlp") == "build"

    def test_classify_train(self):
        assert IntentClassifier.classify("train the model on data") == "train"
        assert IntentClassifier.classify("fit the model") == "train"

    def test_classify_evaluate(self):
        assert IntentClassifier.classify("evaluate the accuracy") == "evaluate"
        assert IntentClassifier.classify("test on test set") == "evaluate"

    def test_classify_save_and_load(self):
        assert IntentClassifier.classify("save it") == "save"
        assert IntentClassifier.classify("load checkpoint") == "load"

    def test_classify_data_pipeline(self):
        assert IntentClassifier.classify("load data from memory") == "load_data"
        assert (
            IntentClassifier.classify("preprocess the dataset by standardizing")
            == "preprocess_data"
        )
        assert IntentClassifier.classify("analyze data to find mean") == "analyze_data"
