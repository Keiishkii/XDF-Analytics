#region [ Class ][ Log ]
class Log:
    MESSAGE_PREFIX: str = "\033[94m[ Message ]\033[0m"
    WARNING_PREFIX: str = "\033[33m[ Warning ]\033[0m"
    ERROR_PREFIX: str = "\033[91m[ Error ]\033[0m"

    @staticmethod
    def message(message: str):
        """
        :param message: The message to be displayed.
        """

        print(f"{Log.MESSAGE_PREFIX} {message}")

    @staticmethod
    def warning(message: str):
        """
        :param message: The message to be displayed.
        """

        print(f"{Log.WARNING_PREFIX} {message}")

    @staticmethod
    def error(message: str):
        """
        :param message: The message to be displayed.
        """

        print(f"{Log.ERROR_PREFIX} {message}")
#endregion