"""
Error Handler

Comprehensive error handling and context capture for
self-healing operations including HTML snapshots and tracebacks.
"""

class ErrorHandler:
    """Handles error capture and context collection for healing."""
    
    def __init__(self):
        self.logger = None
    
    def capture_error(self, exception: Exception, agent_name: str) -> dict:
        """Capture comprehensive error context for healing analysis."""
        from datetime import datetime
        import traceback
        
        return {
            'type': type(exception).__name__,
            'message': str(exception),
            'agent': agent_name,
            'timestamp': datetime.utcnow().isoformat(),
            'traceback': traceback.format_exc(),
            'context': {
                'module': exception.__class__.__module__,
                'args': getattr(exception, 'args', None),
                'kwargs': getattr(exception, 'kwargs', None)
            }
        }