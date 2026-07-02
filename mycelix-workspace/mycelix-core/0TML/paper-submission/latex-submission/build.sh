#!/usr/bin/env bash
#
# Build script for Zero-TrustML IEEE S&P submission
#

set -e  # Exit on error

echo "🔧 Building Zero-TrustML IEEE S&P Submission..."

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "❌ pdflatex not found. Installing texlive..."
    # For NixOS, we need to use nix-shell
    echo "Please run: nix-shell -p texlive.combined.scheme-full"
    exit 1
fi

# Download IEEE template if not present
if [ ! -f "IEEEtran.cls" ]; then
    echo "📥 Downloading IEEE template..."
    wget -q https://www.ctan.org/tex-archive/macros/latex/contrib/IEEEtran/IEEEtran.cls || true

    # If wget fails, create a notice
    if [ ! -f "IEEEtran.cls" ]; then
        echo "⚠️  Please download IEEEtran.cls from:"
        echo "   https://www.ctan.org/tex-archive/macros/latex/contrib/IEEEtran/IEEEtran.cls"
        echo "   Or install: texlive-publishers package"
    fi
fi

# Compile the document
echo "📄 Compiling LaTeX document..."

# First pass
pdflatex -interaction=nonstopmode main.tex || true

# Run bibtex
if [ -f "references.bib" ]; then
    echo "📚 Processing references..."
    bibtex main || true
fi

# Second pass (for references)
pdflatex -interaction=nonstopmode main.tex || true

# Third pass (for cross-references)
pdflatex -interaction=nonstopmode main.tex || true

# Check output
if [ -f "main.pdf" ]; then
    echo "✅ Build successful: main.pdf"

    # Get page count
    if command -v pdfinfo &> /dev/null; then
        pages=$(pdfinfo main.pdf | grep "Pages:" | awk '{print $2}')
        echo "   Pages: $pages"

        if [ "$pages" -gt 13 ]; then
            echo "   ⚠️  WARNING: Exceeds IEEE S&P 13-page limit (excluding references)"
        fi
    fi

    # Get file size
    size=$(du -h main.pdf | cut -f1)
    echo "   Size: $size"

else
    echo "❌ Build failed. Check main.log for errors."
    exit 1
fi

echo ""
echo "🎉 Build complete!"
echo ""
echo "Next steps:"
echo "1. Review main.pdf for formatting"
echo "2. Check page count (target: 12-13 pages + references)"
echo "3. Verify all figures appear correctly"
echo "4. Run anonymization check"
