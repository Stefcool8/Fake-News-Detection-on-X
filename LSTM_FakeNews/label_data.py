class LabelData:
    def __init__(self):
        self.conversion_table = {
            (True, "Agree"): True,
            (True, "Disagree"): False,
            (False, "Agree"): False,
            (False, "Disagree"): True
        }

    def get_truthfulness(self, target, majority_answer):
        return self.conversion_table.get((target, majority_answer))

    @staticmethod
    def convert_truthfulness_to_binary(truthfulness):
        return 1 if truthfulness else 0
