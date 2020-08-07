import enum


class LearningType(enum.Enum):
    ONLINE = 1,
    REPLAY = 2,
    OFF_POLICY = 3,
    COMBINED = 4

    @staticmethod
    def all():
        return [LearningType.ONLINE, LearningType.REPLAY, LearningType.OFF_POLICY, LearningType.COMBINED]

    @staticmethod
    def get_type_by_name(name):
        for learning_type in LearningType.all():
            if learning_type.name.lower() == name.lower():
                return learning_type
        return None
