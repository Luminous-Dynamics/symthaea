#!/bin/bash
# Mycelix Mail - Version Management Script
# =========================================
# Handles version bumping across all packages

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."

# Helper functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Get current version from root package.json
get_current_version() {
    cd "$PROJECT_ROOT"
    if [ -f "package.json" ]; then
        node -p "require('./package.json').version" 2>/dev/null || echo "0.0.0"
    elif [ -f "holochain/client/package.json" ]; then
        node -p "require('./holochain/client/package.json').version" 2>/dev/null || echo "0.0.0"
    else
        echo "0.0.0"
    fi
}

# Bump version based on type
bump_version() {
    local current=$1
    local bump_type=$2

    IFS='.' read -r major minor patch <<< "$current"

    case $bump_type in
        major)
            major=$((major + 1))
            minor=0
            patch=0
            ;;
        minor)
            minor=$((minor + 1))
            patch=0
            ;;
        patch)
            patch=$((patch + 1))
            ;;
        *)
            log_error "Invalid bump type: $bump_type"
            exit 1
            ;;
    esac

    echo "$major.$minor.$patch"
}

# Update version in a JSON file
update_json_version() {
    local file=$1
    local version=$2

    if [ -f "$file" ]; then
        # Use node to update JSON
        node -e "
            const fs = require('fs');
            const pkg = JSON.parse(fs.readFileSync('$file', 'utf8'));
            pkg.version = '$version';
            fs.writeFileSync('$file', JSON.stringify(pkg, null, 2) + '\\n');
        "
        log_success "Updated $file to $version"
    else
        log_warning "File not found: $file"
    fi
}

# Update version in Cargo.toml
update_cargo_version() {
    local file=$1
    local version=$2

    if [ -f "$file" ]; then
        sed -i "s/^version = \"[^\"]*\"/version = \"$version\"/" "$file"
        log_success "Updated $file to $version"
    else
        log_warning "File not found: $file"
    fi
}

# Update all version references
update_all_versions() {
    local version=$1

    log_info "Updating all packages to version $version"

    cd "$PROJECT_ROOT"

    # Root package.json
    if [ -f "package.json" ]; then
        update_json_version "package.json" "$version"
    fi

    # TypeScript client
    update_json_version "holochain/client/package.json" "$version"

    # UI frontend
    if [ -f "ui/frontend/package.json" ]; then
        update_json_version "ui/frontend/package.json" "$version"
    fi

    # Rust workspace
    update_cargo_version "holochain/Cargo.toml" "$version"

    # Individual crates
    for toml in holochain/crates/*/Cargo.toml; do
        update_cargo_version "$toml" "$version"
    done

    # Zome crates
    for toml in holochain/zomes/*/*/Cargo.toml; do
        update_cargo_version "$toml" "$version"
    done

    log_success "All versions updated to $version"
}

# Create git tag
create_tag() {
    local version=$1
    local tag="v$version"

    if git rev-parse "$tag" >/dev/null 2>&1; then
        log_warning "Tag $tag already exists"
        return 1
    fi

    git tag -a "$tag" -m "Release $version"
    log_success "Created tag $tag"
}

# Update changelog
update_changelog() {
    local version=$1
    local date=$(date +%Y-%m-%d)

    if [ -f "$PROJECT_ROOT/CHANGELOG.md" ]; then
        # Replace [Unreleased] with the new version
        sed -i "s/## \[Unreleased\]/## [Unreleased]\n\n## [$version] - $date/" "$PROJECT_ROOT/CHANGELOG.md"

        # Add comparison link
        local prev_version=$(grep -oP '\[\d+\.\d+\.\d+\]' "$PROJECT_ROOT/CHANGELOG.md" | head -2 | tail -1 | tr -d '[]')
        if [ -n "$prev_version" ]; then
            echo "[$version]: https://github.com/luminous-dynamics/mycelix-mail/compare/v$prev_version...v$version" >> "$PROJECT_ROOT/CHANGELOG.md"
        fi

        log_success "Updated CHANGELOG.md"
    fi
}

# Print usage
usage() {
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  current             Show current version"
    echo "  bump <type>         Bump version (major|minor|patch)"
    echo "  set <version>       Set specific version"
    echo "  tag                 Create git tag for current version"
    echo "  release <type>      Bump, update changelog, and create tag"
    echo ""
    echo "Examples:"
    echo "  $0 current"
    echo "  $0 bump patch"
    echo "  $0 set 1.0.0"
    echo "  $0 release minor"
    echo ""
}

# Main
main() {
    case "${1:-}" in
        current)
            get_current_version
            ;;

        bump)
            if [ -z "${2:-}" ]; then
                log_error "Bump type required (major|minor|patch)"
                usage
                exit 1
            fi

            current=$(get_current_version)
            new_version=$(bump_version "$current" "$2")

            log_info "Bumping from $current to $new_version"
            update_all_versions "$new_version"
            ;;

        set)
            if [ -z "${2:-}" ]; then
                log_error "Version required"
                usage
                exit 1
            fi

            # Validate version format
            if ! [[ "$2" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
                log_error "Invalid version format. Use X.Y.Z"
                exit 1
            fi

            log_info "Setting version to $2"
            update_all_versions "$2"
            ;;

        tag)
            version=$(get_current_version)
            create_tag "$version"
            ;;

        release)
            if [ -z "${2:-}" ]; then
                log_error "Bump type required (major|minor|patch)"
                usage
                exit 1
            fi

            # Check for uncommitted changes
            if ! git diff-index --quiet HEAD --; then
                log_error "You have uncommitted changes. Please commit or stash them first."
                exit 1
            fi

            current=$(get_current_version)
            new_version=$(bump_version "$current" "$2")

            log_info "Releasing version $new_version"

            # Update versions
            update_all_versions "$new_version"

            # Update changelog
            update_changelog "$new_version"

            # Commit changes
            git add -A
            git commit -m "Release v$new_version

$(cat <<EOF
## Changes

See CHANGELOG.md for details.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"

            # Create tag
            create_tag "$new_version"

            log_success "Released version $new_version"
            log_info "Run 'git push && git push --tags' to publish"
            ;;

        -h|--help|help)
            usage
            ;;

        *)
            log_error "Unknown command: ${1:-}"
            usage
            exit 1
            ;;
    esac
}

main "$@"
