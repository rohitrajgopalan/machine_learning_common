import enum


class LearningType(enum.Enum):
    ONLINE = 1,
    REPLAY = 2,
    OFF_POLICY = 3

    @staticmethod
    def all():
        return [LearningType.ONLINE, LearningType.REPLAY, LearningType.OFF_POLICY]