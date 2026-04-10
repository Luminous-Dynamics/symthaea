#!/usr/bin/env bash
# interact-window.sh — Focus, resize, and interact with Wayland windows via KWin
#
# Usage:
#   interact-window.sh focus <resourceClass>
#   interact-window.sh maximize <resourceClass>
#   interact-window.sh resize <resourceClass> <width> <height>
#   interact-window.sh info <resourceClass>
#   interact-window.sh click <resourceClass>    # focus + click center
#   interact-window.sh list                     # list all windows

set -euo pipefail
CMD="${1:-list}"
CLASS="${2:-}"

run_kwin_js() {
    local js="$1"
    local SCRIPT=$(mktemp /tmp/kwin-XXXXXX.js)
    echo "$js" > "$SCRIPT"
    local SID=$(qdbus org.kde.KWin /Scripting org.kde.kwin.Scripting.loadScript "$SCRIPT" 2>/dev/null || echo "")
    if [[ -n "$SID" ]]; then
        qdbus org.kde.KWin "/Scripting/Script$SID" org.kde.kwin.Script.run 2>/dev/null || true
    fi
    rm -f "$SCRIPT"
    # Read output from journal
    sleep 0.3
    journalctl -t kwin_wayland --since "2 sec ago" --no-pager 2>/dev/null | grep "INTERACT:" | sed 's/.*INTERACT://'
}

case "$CMD" in
    list)
        run_kwin_js '
var cl = workspace.windowList();
for (var i = 0; i < cl.length; i++) {
    var c = cl[i];
    console.log("INTERACT:" + c.pid + " | " + c.resourceClass + " | " + c.caption + " | " + JSON.stringify(c.frameGeometry));
}'
        ;;
    focus)
        [[ -z "$CLASS" ]] && { echo "Usage: $0 focus <resourceClass>"; exit 1; }
        run_kwin_js "
var cl = workspace.windowList();
for (var i = 0; i < cl.length; i++) {
    var c = cl[i];
    if (c.resourceClass === '$CLASS') {
        c.desktops = [workspace.currentDesktop];
        c.minimized = false;
        workspace.activeWindow = c;
        console.log('INTERACT:focused ' + c.caption);
    }
}"
        ;;
    maximize)
        [[ -z "$CLASS" ]] && { echo "Usage: $0 maximize <resourceClass>"; exit 1; }
        run_kwin_js "
var cl = workspace.windowList();
for (var i = 0; i < cl.length; i++) {
    var c = cl[i];
    if (c.resourceClass === '$CLASS') {
        c.desktops = [workspace.currentDesktop];
        c.minimized = false;
        workspace.activeWindow = c;
        c.frameGeometry = {x: 0, y: 0, width: 1920, height: 1080};
        console.log('INTERACT:maximized ' + c.caption + ' to 1920x1080');
    }
}"
        ;;
    resize)
        W="${3:?width}"
        H="${4:?height}"
        run_kwin_js "
var cl = workspace.windowList();
for (var i = 0; i < cl.length; i++) {
    var c = cl[i];
    if (c.resourceClass === '$CLASS') {
        c.frameGeometry = {x: c.frameGeometry.x, y: c.frameGeometry.y, width: $W, height: $H};
        console.log('INTERACT:resized ' + c.caption + ' to ${W}x${H}');
    }
}"
        ;;
    info)
        [[ -z "$CLASS" ]] && { echo "Usage: $0 info <resourceClass>"; exit 1; }
        run_kwin_js "
var cl = workspace.windowList();
for (var i = 0; i < cl.length; i++) {
    var c = cl[i];
    if (c.resourceClass === '$CLASS') {
        var g = c.frameGeometry;
        console.log('INTERACT:' + c.caption + ' pid=' + c.pid + ' geom=' + g.x + ',' + g.y + ',' + g.width + 'x' + g.height + ' desktop=' + JSON.stringify(c.desktops));
    }
}"
        ;;
    *)
        echo "Usage: $0 <list|focus|maximize|resize|info|click> [resourceClass] [args...]"
        exit 1
        ;;
esac
