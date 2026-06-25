"""Shared tool execution error types."""


class ToolExecutionError(RuntimeError):
    """Raised when a tool fails after authorization and validation."""


class ToolTimeoutError(ToolExecutionError):
    """Raised when a tool exceeds its configured execution timeout."""
