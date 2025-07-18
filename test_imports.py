#!/usr/bin/env python
"""
Test script to verify imports work correctly.
"""

def test_cli_imports():
    """Test imports from CLI module."""
    try:
        from src.cli.ingest import ingest_app
        print("✅ CLI module loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to import CLI module: {e}")
        return False


def test_embeddings_imports():
    """Test imports from embeddings module."""
    try:
        from src.core.embeddings.base import BaseEmbedder, EmbeddingCache
        print("✅ Base embeddings loaded successfully!")
        
        from src.core.embeddings.factory import get_embedder
        print("✅ Embeddings factory loaded successfully!")
        
        from src.core.embeddings.providers.jina import JinaEmbedder
        print("✅ Jina embedder loaded successfully!")
        
        return True
    except Exception as e:
        print(f"❌ Failed to import embeddings module: {e}")
        return False


def main():
    """Run all tests."""
    cli_success = test_cli_imports()
    embeddings_success = test_embeddings_imports()
    
    if cli_success and embeddings_success:
        print("\n✅ All imports successful!")
        return 0
    else:
        print("\n❌ Some imports failed!")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main()) 