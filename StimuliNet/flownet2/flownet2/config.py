from enum import IntEnum, unique

@unique
class config(IntEnum):

      DEFAULT: bin = 0b000
      LOAD_WEIGHTS: bin = 0b001
      SAVE_FLOW: bin = 0b010
      LOSS_EVENT: bin = 0b100

      @staticmethod
      def show() -> None:
          print("config.DEFAULT")
          print("config.LOSS_EVENT")
          print("config.LOAD_WEIGHTS")
          print("config.SAVE_FLOW")
