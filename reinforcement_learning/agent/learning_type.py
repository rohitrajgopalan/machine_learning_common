import enum


class LearningType(enum.Enum):
    ONLINE = 1,
    REPLAY = 2

    @staticmethod
    def all():
        return [LearningType.ONLINE, LearningType.REPLAY]