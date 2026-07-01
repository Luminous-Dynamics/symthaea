# Symthaea Shell Completions for Fish
# E2: Shell Completions
#
# Install: Copy to ~/.config/fish/completions/
#   cp symthaea.fish ~/.config/fish/completions/
#
# Or for system-wide: /usr/share/fish/vendor_completions.d/

# Main commands
set -l commands install add remove uninstall enable disable search rebuild switch status help

# Slash commands
set -l slash_commands /help /quit /exit /q /status /reconnect /history /clear /center /phi /coherence

# Common services
set -l services nginx postgresql mysql redis docker sshd cups bluetooth networkmanager pipewire

# Disable file completion by default
complete -c symthaea -f
complete -c symthaea-shell -f
complete -c symthaea-repl -f

# Main commands
complete -c symthaea -n "not __fish_seen_subcommand_from $commands" -a "install" -d "Install packages"
complete -c symthaea -n "not __fish_seen_subcommand_from $commands" -a "add" -d "Add packages"
complete -c symthaea -n "not __fish_seen_subcommand_from $commands" -a "remove" -d "Remove packages"
complete -c symthaea -n "not __fish_seen_subcommand_from $commands" -a "uninstall" -d "Uninstall packages"
complete -c symthaea -n "not __fish_seen_subcommand_from $commands" -a "enable" -d "Enable service"
complete -c symthaea -n "not __fish_seen_subcommand_from $commands" -a "disable" -d "Disable service"
complete -c symthaea -n "not __fish_seen_subcommand_from $commands" -a "search" -d "Search packages"
complete -c symthaea -n "not __fish_seen_subcommand_from $commands" -a "rebuild" -d "Rebuild NixOS"
complete -c symthaea -n "not __fish_seen_subcommand_from $commands" -a "switch" -d "Switch configuration"
complete -c symthaea -n "not __fish_seen_subcommand_from $commands" -a "status" -d "Show status"
complete -c symthaea -n "not __fish_seen_subcommand_from $commands" -a "help" -d "Show help"

# Slash commands (when input starts with /)
complete -c symthaea -n "string match -q '/*' (commandline -ct)" -a "/help" -d "Show commands"
complete -c symthaea -n "string match -q '/*' (commandline -ct)" -a "/quit" -d "Exit shell"
complete -c symthaea -n "string match -q '/*' (commandline -ct)" -a "/exit" -d "Exit shell"
complete -c symthaea -n "string match -q '/*' (commandline -ct)" -a "/status" -d "Connection status"
complete -c symthaea -n "string match -q '/*' (commandline -ct)" -a "/reconnect" -d "Reconnect"
complete -c symthaea -n "string match -q '/*' (commandline -ct)" -a "/history" -d "Command history"
complete -c symthaea -n "string match -q '/*' (commandline -ct)" -a "/clear" -d "Clear output"
complete -c symthaea -n "string match -q '/*' (commandline -ct)" -a "/center" -d "Center Phi"

# Package completions for install/add
complete -c symthaea -n "__fish_seen_subcommand_from install add" -a "(nix-env -qaP --no-name 2>/dev/null | head -50)"

# Installed packages for remove/uninstall
complete -c symthaea -n "__fish_seen_subcommand_from remove uninstall" -a "(nix-env -q 2>/dev/null)"

# Service completions
complete -c symthaea -n "__fish_seen_subcommand_from enable disable" -a "nginx" -d "Web server"
complete -c symthaea -n "__fish_seen_subcommand_from enable disable" -a "postgresql" -d "PostgreSQL database"
complete -c symthaea -n "__fish_seen_subcommand_from enable disable" -a "mysql" -d "MySQL database"
complete -c symthaea -n "__fish_seen_subcommand_from enable disable" -a "redis" -d "Redis cache"
complete -c symthaea -n "__fish_seen_subcommand_from enable disable" -a "docker" -d "Docker runtime"
complete -c symthaea -n "__fish_seen_subcommand_from enable disable" -a "sshd" -d "SSH server"
complete -c symthaea -n "__fish_seen_subcommand_from enable disable" -a "cups" -d "Print service"
complete -c symthaea -n "__fish_seen_subcommand_from enable disable" -a "bluetooth" -d "Bluetooth"
complete -c symthaea -n "__fish_seen_subcommand_from enable disable" -a "networkmanager" -d "Network"
complete -c symthaea -n "__fish_seen_subcommand_from enable disable" -a "pipewire" -d "Audio"

# Rebuild options
complete -c symthaea -n "__fish_seen_subcommand_from rebuild switch" -l dry-run -d "Show what would be built"
complete -c symthaea -n "__fish_seen_subcommand_from rebuild switch" -l rollback -d "Rollback to previous"
complete -c symthaea -n "__fish_seen_subcommand_from rebuild switch" -l test -d "Build without making bootable"
complete -c symthaea -n "__fish_seen_subcommand_from rebuild switch" -l build-host -d "Build on remote host"
complete -c symthaea -n "__fish_seen_subcommand_from rebuild switch" -l target-host -d "Activate on remote host"

# symthaea-shell and symthaea-repl (same completions)
for cmd in symthaea-shell symthaea-repl
    complete -c $cmd -n "not __fish_seen_subcommand_from $commands" -a "install" -d "Install packages"
    complete -c $cmd -n "not __fish_seen_subcommand_from $commands" -a "remove" -d "Remove packages"
    complete -c $cmd -n "not __fish_seen_subcommand_from $commands" -a "enable" -d "Enable service"
    complete -c $cmd -n "not __fish_seen_subcommand_from $commands" -a "disable" -d "Disable service"
    complete -c $cmd -n "not __fish_seen_subcommand_from $commands" -a "search" -d "Search packages"
    complete -c $cmd -n "not __fish_seen_subcommand_from $commands" -a "rebuild" -d "Rebuild NixOS"
    complete -c $cmd -n "not __fish_seen_subcommand_from $commands" -a "help" -d "Show help"
end

# symthaea-service completions
complete -c symthaea-service -f
complete -c symthaea-service -s s -l socket -d "Socket path" -r
complete -c symthaea-service -s i -l interval -d "Metrics interval (ms)" -a "100 250 500 1000 2000"
complete -c symthaea-service -s h -l help -d "Show help"

# symthaea-gui completions
complete -c symthaea-gui -f
complete -c symthaea-gui -l socket -d "Socket path" -r
complete -c symthaea-gui -l theme -d "Color theme" -a "dark light system"
complete -c symthaea-gui -l help -d "Show help"
