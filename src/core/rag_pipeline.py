# Backwards compatibility — this file has been replaced by the architecture system.
# Use src.core.runner.PipelineRunner instead.
#
# Migration:
#   Old: from src.core.rag_pipeline import RAGPipeline
#        pipeline = RAGPipeline(verbose=True)
#        response = pipeline.run(query)
#
#   New: from src.core.runner import PipelineRunner
#        runner = PipelineRunner("naive", verbose=True)
#        response = runner.run(query)

from src.core.runner import PipelineRunner


class RAGPipeline:
    """
    Backwards-compatible wrapper around PipelineRunner.
    
    Deprecated: Use PipelineRunner directly for access to all architectures.
    """

    def __init__(self, verbose: bool = False):
        import warnings
        warnings.warn(
            "RAGPipeline is deprecated. Use 'from src.core.runner import PipelineRunner' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._runner = PipelineRunner(architecture_name="naive", verbose=verbose)

    def run(self, query: str, session_id: str = "default") -> str:
        return self._runner.run(query, session_id)

    def check_health(self):
        self._runner.check_health()