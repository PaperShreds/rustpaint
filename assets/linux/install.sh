#!/bin/sh
# Gives RustPaint its name and icon in the application menu, the dock and the
# window title bar on Linux. Desktops find those through a .desktop file whose
# name matches the window's app id ("rustpaint"), so a build run straight from
# `cargo run` shows a generic icon until this has been run once.
#
#   assets/linux/install.sh [path/to/rustpaint]   install (default: the release
#                                                 build next to this checkout,
#                                                 or `rustpaint` on the PATH)
#   assets/linux/install.sh --uninstall
set -eu

here=$(cd "$(dirname "$0")" && pwd)
data=${XDG_DATA_HOME:-$HOME/.local/share}
desktop=$data/applications/rustpaint.desktop
png=$data/icons/hicolor/256x256/apps/rustpaint.png
svg=$data/icons/hicolor/scalable/apps/rustpaint.svg

refresh() {
    command -v update-desktop-database >/dev/null && update-desktop-database "$data/applications" || true
    command -v gtk-update-icon-cache >/dev/null && gtk-update-icon-cache -q -t "$data/icons/hicolor" || true
}

if [ "${1:-}" = "--uninstall" ]; then
    rm -f "$desktop" "$png" "$svg"
    refresh
    echo "Removed RustPaint's launcher and icon."
    exit 0
fi

exe=${1:-}
if [ -z "$exe" ]; then
    if [ -x "$here/../../target/release/rustpaint" ]; then
        exe=$here/../../target/release/rustpaint
    else
        exe=$(command -v rustpaint || true)
    fi
fi
if [ -z "$exe" ] || [ ! -x "$exe" ]; then
    echo "Build RustPaint first (cargo build --release), or pass its path." >&2
    exit 1
fi
exe=$(cd "$(dirname "$exe")" && pwd)/$(basename "$exe")

mkdir -p "$(dirname "$desktop")" "$(dirname "$png")" "$(dirname "$svg")"
sed "s|^Exec=.*|Exec=$exe|" "$here/rustpaint.desktop" > "$desktop"
cp "$here/../logo/icon.png" "$png"
cp "$here/../logo/logo.svg" "$svg"
refresh
echo "Installed the launcher for $exe."
echo "Restart RustPaint (log out and in if the icon doesn't change)."
