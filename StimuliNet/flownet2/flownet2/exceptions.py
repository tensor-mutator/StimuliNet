class NotTrainableError(Exception):

      def __init__(self, msg: str) -> None:
          super(NotTrainableError, self).__init__(msg)

class InvalidHyperparameterError(Exception):

      def __init__(self, msg: str) -> None:
          super(InvalidHyperparameterError, self).__init__(msg)
