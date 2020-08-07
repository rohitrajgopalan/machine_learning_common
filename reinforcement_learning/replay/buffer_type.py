import enum


class BufferType(enum.Enum):
    BASIC = 1,
    PRIORITIZED = 2,
    RANDOMIZED = 3

    @staticmethod
    def all():
        return [BufferType.BASIC, BufferType.PRIORITIZED, BufferType.RANDOMIZED]

    @staticmethod
    def get_type_by_name(name):
        for buffer_type in BufferType.all():
            if buffer_type.name.lower() == name.lower():
                return buffer_type
        return None
