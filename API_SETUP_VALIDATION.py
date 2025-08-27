#!/usr/bin/env python3
"""
Validate aisitools API setup on new VM
"""

import os
import sys

def validate_api_setup():
    """Test the complete aisitools -> Claude API chain"""
    
    print("🔍 VALIDATING AISI API SETUP")
    print("=" * 40)
    
    # Check environment variables
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    anthropic_base = os.environ.get("ANTHROPIC_BASE_URL") 
    
    if not anthropic_key:
        print("❌ ANTHROPIC_API_KEY not set")
        return False
        
    if not anthropic_base:
        print("❌ ANTHROPIC_BASE_URL not set")
        return False
        
    print(f"✅ ANTHROPIC_API_KEY: {anthropic_key[:50]}...")
    print(f"✅ ANTHROPIC_BASE_URL: {anthropic_base}")
    
    # Test aisitools import
    try:
        from aisitools.api_key import get_api_key_for_proxy
        print("✅ aisitools imported successfully")
    except ImportError as e:
        print(f"❌ aisitools import failed: {e}")
        print("Run: pip install --break-system-packages git+ssh://git@github.com/AI-Safety-Institute/aisi-inspect-tools")
        return False
        
    # Test key conversion
    try:
        proxy_key = get_api_key_for_proxy(anthropic_key)
        print(f"✅ Proxy key generated: {proxy_key[:30]}...")
    except Exception as e:
        print(f"❌ Key conversion failed: {e}")
        return False
        
    # Test anthropic client
    try:
        import anthropic
        client = anthropic.Anthropic(
            api_key=proxy_key,
            base_url=anthropic_base
        )
        print("✅ Anthropic client created")
    except Exception as e:
        print(f"❌ Anthropic client failed: {e}")
        return False
        
    # Test actual API call
    try:
        print("🧪 Testing actual API call...")
        response = client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=20,
            messages=[{"role": "user", "content": "Say 'API test successful'"}]
        )
        result = response.content[0].text
        print(f"✅ API Response: {result}")
        
        if "successful" in result.lower():
            print("\n🎉 COMPLETE SUCCESS - Ready for contextual chunking!")
            return True
        else:
            print("⚠️  API responded but unexpected content")
            return False
            
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return False

if __name__ == "__main__":
    success = validate_api_setup()
    if success:
        print("\n✅ All systems go for contextual chunking!")
        sys.exit(0)
    else:
        print("\n❌ Setup validation failed - fix issues before proceeding")
        sys.exit(1)