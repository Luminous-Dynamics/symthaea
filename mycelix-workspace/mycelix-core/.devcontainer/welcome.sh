#!/bin/bash
# Mycelix Welcome Message (shown once per terminal session)

WELCOME_FLAG="/tmp/.mycelix_welcomed"

if [ ! -f "$WELCOME_FLAG" ]; then
    echo ""
    echo "  ╔══════════════════════════════════════════════════════════════╗"
    echo "  ║                                                              ║"
    echo "  ║   ███╗   ███╗██╗   ██╗ ██████╗███████╗██╗     ██╗██╗  ██╗   ║"
    echo "  ║   ████╗ ████║╚██╗ ██╔╝██╔════╝██╔════╝██║     ██║╚██╗██╔╝   ║"
    echo "  ║   ██╔████╔██║ ╚████╔╝ ██║     █████╗  ██║     ██║ ╚███╔╝    ║"
    echo "  ║   ██║╚██╔╝██║  ╚██╔╝  ██║     ██╔══╝  ██║     ██║ ██╔██╗    ║"
    echo "  ║   ██║ ╚═╝ ██║   ██║   ╚██████╗███████╗███████╗██║██╔╝ ██╗   ║"
    echo "  ║   ╚═╝     ╚═╝   ╚═╝    ╚═════╝╚══════╝╚══════╝╚═╝╚═╝  ╚═╝   ║"
    echo "  ║                                                              ║"
    echo "  ║         Byzantine-Resistant Federated Learning               ║"
    echo "  ║              The Future of Decentralized AI                  ║"
    echo "  ║                                                              ║"
    echo "  ╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  Quick Commands:"
    echo "  ───────────────────────────────────────────────────────────────"
    echo "  demo         Run interactive FL demonstration"
    echo "  fl-demo      Byzantine resistance simulation"
    echo "  cargo test   Run the test suite (255+ tests)"
    echo "  cargo build  Build all components"
    echo ""
    echo "  Documentation: docs/GETTING_STARTED.md"
    echo "  ───────────────────────────────────────────────────────────────"
    echo ""
    touch "$WELCOME_FLAG"
fi
