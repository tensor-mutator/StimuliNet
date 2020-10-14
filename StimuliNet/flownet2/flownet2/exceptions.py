class NotTrainableError(Exception):

      def __init__(self, msg: str) -> None:
          super(NotTrainableError, self).__init__(msg)

class ScheduleNotFoundError(Exception):

      def __init__(self, msg: str) -> None:
          super(ScheduleNotFoundError, self).__init__(msg)

class WeightsNotFoundError(Exception):

      def __init__(self, msg: str) -> None:
          super(WeightsNotFoundError, self).__init__(msg)
