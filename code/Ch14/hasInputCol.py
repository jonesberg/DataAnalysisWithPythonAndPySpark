class HasInputCols(Params):
    """Mixin for param inputCols: input column names."""

    inputCols = Param(  # <1>
        Params._dummy(),
        "inputCols", "input column names.",
        typeConverter=TypeConverters.toListString,
    )

    def __init__(self):
        super(HasInputCols, self).__init__()

    def getInputCols(self):
        """Gets the value of inputCols or its default value. """
        return self.getOrDefault(self.inputCols)
