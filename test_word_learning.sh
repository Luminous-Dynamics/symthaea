#!/usr/bin/env bash
# Test WordLearner integration with slang

echo "Testing Symthaea WordLearner Integration"
echo "========================================="
echo ""
echo "Sending slang-heavy input to test word learning..."
echo ""

# Test with slang input
cat <<EOF | cargo run --quiet --bin symthaea_chat 2>/dev/null | grep -A 100 "Symthaea"
That's lowkey fire bruh
/learn
/quit
EOF

echo ""
echo "========================================="
echo "Test complete!"
