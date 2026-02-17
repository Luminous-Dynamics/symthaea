#!/bin/bash
#
# Mycelix Mail Documentation Generator
#
# Generates comprehensive API documentation for:
# - Rust/Holochain zomes (cargo doc)
# - TypeScript SDK (TypeDoc)
# - Creates unified index.html
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Documentation directories
DOCS_DIR="$PROJECT_ROOT/docs"
API_DOCS_DIR="$DOCS_DIR/api"
RUST_DOCS_DIR="$API_DOCS_DIR/rust"
TS_DOCS_DIR="$API_DOCS_DIR/typescript"

# Source directories
HOLOCHAIN_DIR="$PROJECT_ROOT/holochain"
SDK_TS_DIR="$PROJECT_ROOT/sdk/typescript"

# Flags
RUST_ONLY=false
TS_ONLY=false
CLEAN=false
OPEN_BROWSER=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --rust-only)
            RUST_ONLY=true
            shift
            ;;
        --ts-only)
            TS_ONLY=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --open)
            OPEN_BROWSER=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --rust-only    Generate only Rust documentation"
            echo "  --ts-only      Generate only TypeScript documentation"
            echo "  --clean        Clean existing docs before generating"
            echo "  --open         Open documentation in browser after generation"
            echo "  -h, --help     Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Print header
echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}   Mycelix Mail Documentation Generator${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""

# Clean if requested
if [ "$CLEAN" = true ]; then
    echo -e "${YELLOW}Cleaning existing documentation...${NC}"
    rm -rf "$RUST_DOCS_DIR"/* 2>/dev/null || true
    rm -rf "$TS_DOCS_DIR"/* 2>/dev/null || true
    echo -e "${GREEN}Cleaned.${NC}"
    echo ""
fi

# Ensure directories exist
mkdir -p "$RUST_DOCS_DIR"
mkdir -p "$TS_DOCS_DIR"

# Generate Rust documentation
generate_rust_docs() {
    echo -e "${BLUE}Generating Rust/Holochain API documentation...${NC}"

    if [ ! -d "$HOLOCHAIN_DIR" ]; then
        echo -e "${RED}Holochain directory not found: $HOLOCHAIN_DIR${NC}"
        return 1
    fi

    cd "$HOLOCHAIN_DIR"

    # Generate documentation for all workspace members
    echo -e "${YELLOW}Running cargo doc for workspace...${NC}"

    # Set environment for documentation
    export RUSTDOCFLAGS="--enable-index-page -Zunstable-options"

    # Generate docs without dependencies for cleaner output
    if cargo doc --no-deps --workspace --document-private-items 2>/dev/null; then
        # Copy generated docs to API docs directory
        if [ -d "$HOLOCHAIN_DIR/target/doc" ]; then
            echo -e "${YELLOW}Copying Rust docs to $RUST_DOCS_DIR...${NC}"
            cp -r "$HOLOCHAIN_DIR/target/doc/"* "$RUST_DOCS_DIR/"
            echo -e "${GREEN}Rust documentation generated successfully!${NC}"
        else
            echo -e "${RED}Rust docs not found in target/doc${NC}"
            return 1
        fi
    else
        # Fallback without unstable flags
        echo -e "${YELLOW}Retrying without unstable options...${NC}"
        unset RUSTDOCFLAGS
        cargo doc --no-deps --workspace 2>/dev/null || {
            echo -e "${YELLOW}Generating docs for individual zomes...${NC}"

            # Generate docs for each coordinator zome
            for zome_dir in zomes/*/coordinator; do
                if [ -d "$zome_dir" ]; then
                    zome_name=$(basename "$(dirname "$zome_dir")")
                    echo -e "  Documenting $zome_name..."
                    (cd "$zome_dir" && cargo doc --no-deps 2>/dev/null) || true
                fi
            done
        }

        # Copy whatever docs were generated
        if [ -d "$HOLOCHAIN_DIR/target/doc" ]; then
            cp -r "$HOLOCHAIN_DIR/target/doc/"* "$RUST_DOCS_DIR/" 2>/dev/null || true
            echo -e "${GREEN}Rust documentation generated.${NC}"
        fi
    fi

    cd "$PROJECT_ROOT"
    echo ""
}

# Generate TypeScript documentation
generate_ts_docs() {
    echo -e "${BLUE}Generating TypeScript SDK documentation...${NC}"

    if [ ! -d "$SDK_TS_DIR" ]; then
        echo -e "${RED}TypeScript SDK directory not found: $SDK_TS_DIR${NC}"
        return 1
    fi

    cd "$SDK_TS_DIR"

    # Check if TypeDoc is configured
    if [ ! -f "typedoc.json" ] && ! grep -q "typedoc" package.json 2>/dev/null; then
        echo -e "${YELLOW}TypeDoc not configured. Setting up...${NC}"

        # Install TypeDoc if not present
        if ! npm list typedoc >/dev/null 2>&1; then
            echo -e "${YELLOW}Installing TypeDoc...${NC}"
            npm install --save-dev typedoc typedoc-plugin-markdown 2>/dev/null || true
        fi
    fi

    # Check if typedoc is available
    if npx typedoc --version >/dev/null 2>&1; then
        echo -e "${YELLOW}Running TypeDoc...${NC}"

        # Generate documentation
        npx typedoc \
            --out "$TS_DOCS_DIR" \
            --entryPoints src/index.ts src/holochain/index.ts \
            --entryPointStrategy expand \
            --name "Mycelix Mail TypeScript SDK" \
            --readme "$SDK_TS_DIR/README.md" \
            --includeVersion \
            --excludePrivate \
            --excludeProtected \
            --plugin typedoc-plugin-markdown \
            --categorizeByGroup true \
            --sort alphabetical \
            --navigationLinks '{"GitHub": "https://github.com/luminous-dynamics/mycelix-mail"}' \
            2>/dev/null || {
                # Fallback to basic TypeDoc
                echo -e "${YELLOW}Retrying with basic TypeDoc config...${NC}"
                npx typedoc \
                    --out "$TS_DOCS_DIR" \
                    --entryPoints src/index.ts \
                    --name "Mycelix Mail TypeScript SDK" \
                    --excludePrivate \
                    2>/dev/null || true
            }

        echo -e "${GREEN}TypeScript documentation generated successfully!${NC}"
    else
        echo -e "${YELLOW}TypeDoc not available. Creating basic documentation...${NC}"

        # Create a basic documentation file
        cat > "$TS_DOCS_DIR/index.html" << 'TSEOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mycelix Mail TypeScript SDK Documentation</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #333; }
        code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }
        pre { background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }
    </style>
</head>
<body>
    <h1>Mycelix Mail TypeScript SDK</h1>
    <p>To generate full TypeDoc documentation, install TypeDoc:</p>
    <pre><code>npm install --save-dev typedoc
npm run docs</code></pre>
    <p>See the <a href="../../guides/getting-started.md">Getting Started Guide</a> for usage examples.</p>
</body>
</html>
TSEOF
    fi

    cd "$PROJECT_ROOT"
    echo ""
}

# Create unified index.html
create_index() {
    echo -e "${BLUE}Creating documentation index...${NC}"

    cat > "$API_DOCS_DIR/index.html" << 'INDEXEOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mycelix Mail API Documentation</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            max-width: 900px;
            width: 100%;
            padding: 40px;
        }
        header {
            text-align: center;
            margin-bottom: 40px;
        }
        h1 {
            font-size: 2.5rem;
            color: #2d3748;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #718096;
            font-size: 1.1rem;
        }
        .docs-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .doc-card {
            background: #f7fafc;
            border-radius: 12px;
            padding: 24px;
            text-decoration: none;
            color: inherit;
            transition: transform 0.2s, box-shadow 0.2s;
            border: 2px solid transparent;
        }
        .doc-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.1);
            border-color: #667eea;
        }
        .doc-card h2 {
            color: #2d3748;
            font-size: 1.3rem;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .doc-card .icon {
            font-size: 1.5rem;
        }
        .doc-card p {
            color: #718096;
            font-size: 0.95rem;
        }
        .doc-card.rust { border-left: 4px solid #dea584; }
        .doc-card.typescript { border-left: 4px solid #3178c6; }
        .doc-card.guides { border-left: 4px solid #38a169; }
        .quick-links {
            background: #edf2f7;
            border-radius: 12px;
            padding: 24px;
        }
        .quick-links h3 {
            color: #2d3748;
            margin-bottom: 16px;
        }
        .quick-links ul {
            list-style: none;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 8px;
        }
        .quick-links a {
            color: #667eea;
            text-decoration: none;
            padding: 8px 12px;
            border-radius: 6px;
            display: block;
            transition: background 0.2s;
        }
        .quick-links a:hover {
            background: white;
        }
        footer {
            text-align: center;
            margin-top: 40px;
            color: #a0aec0;
            font-size: 0.9rem;
        }
        footer a {
            color: #667eea;
            text-decoration: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Mycelix Mail API Documentation</h1>
            <p class="subtitle">Decentralized, trust-based email on Holochain</p>
        </header>

        <div class="docs-grid">
            <a href="rust/index.html" class="doc-card rust">
                <h2><span class="icon">&#129408;</span> Rust/Holochain API</h2>
                <p>Documentation for Holochain zomes including messages, contacts, trust, federation, and more. Generated with cargo doc.</p>
            </a>

            <a href="typescript/index.html" class="doc-card typescript">
                <h2><span class="icon">&#128218;</span> TypeScript SDK</h2>
                <p>Official TypeScript/JavaScript SDK for REST API and direct Holochain client integration. Generated with TypeDoc.</p>
            </a>

            <a href="../guides/getting-started.md" class="doc-card guides">
                <h2><span class="icon">&#128640;</span> Getting Started</h2>
                <p>Quick start guide for developers. Learn how to set up your development environment and build your first integration.</p>
            </a>
        </div>

        <div class="quick-links">
            <h3>Quick Links</h3>
            <ul>
                <li><a href="../guides/architecture.md">Architecture Overview</a></li>
                <li><a href="../guides/deployment.md">Deployment Guide</a></li>
                <li><a href="rust/mail_messages_coordinator/index.html">Messages Zome API</a></li>
                <li><a href="rust/mail_trust_coordinator/index.html">Trust Zome API</a></li>
                <li><a href="rust/mail_contacts_coordinator/index.html">Contacts Zome API</a></li>
                <li><a href="rust/mail_federation_coordinator/index.html">Federation Zome API</a></li>
            </ul>
        </div>

        <footer>
            <p>Generated on <span id="date"></span></p>
            <p>
                <a href="https://github.com/luminous-dynamics/mycelix-mail">GitHub</a> |
                <a href="https://mycelix.mail">Website</a>
            </p>
        </footer>
    </div>

    <script>
        document.getElementById('date').textContent = new Date().toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
    </script>
</body>
</html>
INDEXEOF

    echo -e "${GREEN}Documentation index created at $API_DOCS_DIR/index.html${NC}"
    echo ""
}

# Main execution
main() {
    if [ "$TS_ONLY" = false ]; then
        generate_rust_docs
    fi

    if [ "$RUST_ONLY" = false ]; then
        generate_ts_docs
    fi

    create_index

    echo -e "${GREEN}==========================================${NC}"
    echo -e "${GREEN}   Documentation generation complete!${NC}"
    echo -e "${GREEN}==========================================${NC}"
    echo ""
    echo -e "Documentation available at:"
    echo -e "  ${BLUE}$API_DOCS_DIR/index.html${NC}"
    echo ""
    echo -e "Individual docs:"
    echo -e "  Rust:       ${BLUE}$RUST_DOCS_DIR/index.html${NC}"
    echo -e "  TypeScript: ${BLUE}$TS_DOCS_DIR/index.html${NC}"
    echo ""

    if [ "$OPEN_BROWSER" = true ]; then
        echo -e "${YELLOW}Opening documentation in browser...${NC}"
        if command -v xdg-open >/dev/null 2>&1; then
            xdg-open "$API_DOCS_DIR/index.html"
        elif command -v open >/dev/null 2>&1; then
            open "$API_DOCS_DIR/index.html"
        else
            echo -e "${YELLOW}Could not detect browser opener. Please open manually.${NC}"
        fi
    fi
}

main
