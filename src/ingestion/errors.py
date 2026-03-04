class IngestionError(Exception):
    """Raised when GPX ingestion fails. Pipeline catches this one type.

    Error codes:
        EMPTY_TRACK  — zero trackpoints in GPX file
        PARSE_ERROR  — malformed XML or unreadable file
    """

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"
