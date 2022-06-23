from enum import Enum


class AnswerLengthComparison(Enum):
    EQUAL = 0
    NONE = 1
    TOO_LONG = 2
    TOO_SHORT = 3
