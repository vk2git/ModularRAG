#!/usr/bin/env python3
"""
Quick verification script for ModularRAG v2.

Run with: uv run python verify.py
"""

import sys

errors = []

# 1. Config loader
try:
    from src.utils.config_loader import load_config, load_architecture_config, get_active_architecture
    config = load_config()
    active = get_active_architecture(config)
    print(f"✅ Config loader OK — active architecture: {active}")
except Exception as e:
    errors.append(f"❌ Config loader: {e}")
    print(errors[-1])

# 2. Registry
try:
    from src.core.registry import ArchitectureRegistry
    registry = ArchitectureRegistry()
    archs = registry.list_architectures()
    print(f"✅ Registry OK — {len(archs)} architectures found:")
    for a in archs:
        print(f"   {a['name']:15s} {a['status']}")
except Exception as e:
    errors.append(f"❌ Registry: {e}")
    print(errors[-1])

# 3. Architecture configs
try:
    for name in ["naive", "advanced", "corrective", "self_rag", "agentic", "adaptive", "graph_rag"]:
        cfg = load_architecture_config(name)
        status = "✅" if cfg else "⚠️ empty"
        print(f"   {name:15s} config: {status}")
except Exception as e:
    errors.append(f"❌ Architecture configs: {e}")
    print(errors[-1])

# 4. Component imports
try:
    from src.core.components import LLMFactory, EmbeddingFactory, VectorStoreFactory, MemoryFactory
    from src.core.components.retriever import RetrieverFactory
    from src.core.components.reranker import RerankerFactory
    from src.core.components.web_search import WebSearchTool
    print("✅ All component imports OK")
except Exception as e:
    errors.append(f"❌ Component imports: {e}")
    print(errors[-1])

# 5. Backwards compatibility
try:
    from src.core.llm import LLMFactory as LF1
    from src.core.embedding import EmbeddingFactory as EF1
    from src.core.vector_store import VectorStoreFactory as VF1
    from src.core.memory import MemoryFactory as MF1
    print("✅ Backwards compatibility OK")
except Exception as e:
    errors.append(f"❌ Backwards compatibility: {e}")
    print(errors[-1])

# 6. Runner import
try:
    from src.core.runner import PipelineRunner
    print("✅ PipelineRunner import OK")
except Exception as e:
    errors.append(f"❌ PipelineRunner: {e}")
    print(errors[-1])

# Summary
print(f"\n{'='*40}")
if errors:
    print(f"FAILED: {len(errors)} error(s)")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED ✅")
    sys.exit(0)
