#!/usr/bin/env bash
# Symthaea Shell Completions for Bash
# E2: Shell Completions
#
# Install: source symthaea.bash
# Or add to ~/.bashrc: source /path/to/symthaea.bash
#
# For NixOS: Add to environment.interactiveShellInit

_symthaea_completions() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # Main commands
    local commands="install remove enable disable search rebuild status help"

    # Slash commands
    local slash_commands="/help /quit /exit /status /reconnect /history /clear /center /phi /coherence"

    case "${prev}" in
        symthaea|symthaea-shell)
            COMPREPLY=( $(compgen -W "${commands}" -- "${cur}") )
            return 0
            ;;
        install|add)
            # Package completion - query nixpkgs
            if command -v nix-env &> /dev/null; then
                local packages
                packages=$(nix-env -qaP --no-name 2>/dev/null | awk '{print $1}' | head -50)
                COMPREPLY=( $(compgen -W "${packages}" -- "${cur}") )
            fi
            return 0
            ;;
        remove|uninstall)
            # Installed packages
            if command -v nix-env &> /dev/null; then
                local installed
                installed=$(nix-env -q 2>/dev/null)
                COMPREPLY=( $(compgen -W "${installed}" -- "${cur}") )
            fi
            return 0
            ;;
        enable|disable)
            # Common services
            local services="nginx postgresql mysql redis docker sshd cups printing bluetooth networkmanager"
            COMPREPLY=( $(compgen -W "${services}" -- "${cur}") )
            return 0
            ;;
        search)
            # No completion for search query
            return 0
            ;;
        rebuild|switch)
            local rebuild_opts="--dry-run --rollback --test --build-host --target-host"
            COMPREPLY=( $(compgen -W "${rebuild_opts}" -- "${cur}") )
            return 0
            ;;
    esac

    # Slash command completion
    if [[ "${cur}" == /* ]]; then
        COMPREPLY=( $(compgen -W "${slash_commands}" -- "${cur}") )
        return 0
    fi

    # Default: show all commands
    COMPREPLY=( $(compgen -W "${commands}" -- "${cur}") )
}

# Register completions
complete -F _symthaea_completions symthaea
complete -F _symthaea_completions symthaea-shell
complete -F _symthaea_completions symthaea-repl

# Service completions
_symthaea_service_completions() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    case "${prev}" in
        symthaea-service)
            COMPREPLY=( $(compgen -W "--socket --interval --help" -- "${cur}") )
            return 0
            ;;
        --socket|-s)
            # Socket path completion
            COMPREPLY=( $(compgen -f -- "${cur}") )
            return 0
            ;;
        --interval|-i)
            # Common intervals
            COMPREPLY=( $(compgen -W "100 250 500 1000 2000" -- "${cur}") )
            return 0
            ;;
    esac
}

complete -F _symthaea_service_completions symthaea-service
