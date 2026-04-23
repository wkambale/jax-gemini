class IntentClassifier:
    @staticmethod
    def classify(prompt: str) -> str:
        # A simple keyword-based intent classifier for now
        p = prompt.lower()
        if "train" in p or "fit" in p:
            return "train"
        elif "eval" in p or "test" in p:
            return "evaluate"
        elif "save" in p or "export" in p:
            return "save"
        elif "load" in p or "import" in p:
            return "load"
        elif "build" in p or "create" in p or "make" in p or "add" in p or "change" in p:
            return "build"
        else:
            return "build"  # Default to build
