class IntentClassifier:
    @staticmethod
    def classify(prompt: str) -> str:
        # A simple keyword-based intent classifier for now
        p = prompt.lower()
        if "preprocess" in p or "normalize" in p or "clean" in p or "reshape" in p:
            return "preprocess_data"
        elif "analyze" in p or "explore" in p or "stats" in p or "stat" in p:
            return "analyze_data"
        elif "load data" in p or "read data" in p or "dataset" in p or "generate data" in p:
            return "load_data"
        elif "train" in p or "fit" in p:
            return "train"
        elif "eval" in p or "test" in p:
            return "evaluate"
        elif "save" in p or "export" in p:
            return "save"
        elif "load" in p and "data" not in p and "dataset" not in p:
            return "load"
        elif "build" in p or "create" in p or "make" in p or "add" in p or "change" in p:
            return "build"
        else:
            return "build"  # Default to build
