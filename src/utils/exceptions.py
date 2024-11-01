""" Errors module
"""


class UnknownStructureException(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class NotCallableElementException(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class WrongShapeException(Exception):
    def __init__(self, *args):
        super().__init__(*args)
