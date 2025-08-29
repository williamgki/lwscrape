#!/bin/bash
# Quick setup script for new VM - contextual chunking

echo "🚀 Setting up new VM for LessWrong contextual chunking..."

# Create working directory
mkdir -p /home/ubuntu/LW_scrape
cd /home/ubuntu/LW_scrape

# Setup Python environment
python3 -m venv venv
source venv/bin/activate

# Verify environment variables FIRST
if [ -z "$ANTHROPIC_API_KEY" ] || [ -z "$ANTHROPIC_BASE_URL" ]; then
    echo "❌ MISSING: Set ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL environment variables"
    echo "export ANTHROPIC_API_KEY='aws-secretsmanager://arn:aws:secretsmanager:...'"
    echo "export ANTHROPIC_BASE_URL='https://anthropic-proxy.i.apps.ai-safety-institute.org.uk'"
    exit 1
fi

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install pandas tiktoken anthropic

# CRITICAL: Install aisitools for Claude API proxy
echo "🔐 Installing aisitools (requires GitHub SSH access)..."
pip install --break-system-packages git+ssh://git@github.com/AI-Safety-Institute/aisi-inspect-tools

# Test aisitools integration
echo "🧪 Testing aisitools API integration..."
python -c "
from aisitools.api_key import get_api_key_for_proxy
import os
try:
    key = os.environ.get('ANTHROPIC_API_KEY')
    proxy_key = get_api_key_for_proxy(key)
    print('✅ aisitools working:', proxy_key[:20] + '...')
except Exception as e:
    print('❌ aisitools test failed:', e)
    exit(1)
"

echo "✅ Environment setup complete"
echo "📋 Next steps:"
echo "   1. Copy document_store.parquet (217MB)"
echo "   2. Copy enhanced_contextual_chunker.py and production_chunking_pipeline.py"
echo "   3. Modify for your document partition (see CHUNKING_HANDOFF_GUIDE.md)"
echo "   4. Launch: python production_chunking_pipeline.py"

echo "🎯 Expected performance: 1,100+ docs/hour with powerful VM"