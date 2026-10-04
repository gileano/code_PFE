#!/usr/bin/env bash
# Build the survey paper locally with the same TeX Live distribution Overleaf uses,
# via the texlive/texlive Docker image (no local TeX install needed).
#
# Usage:
#   ./build.sh          build once -> main.pdf (aux files kept in build/)
#   ./build.sh watch    rebuild automatically whenever a source file changes
#   ./build.sh clean    remove build/ and main.pdf
set -euo pipefail

IMAGE="${TEXLIVE_IMAGE:-texlive/texlive:latest}"
cd "$(dirname "$0")"

run_tex() {
    docker run --rm -i \
        -u "$(id -u):$(id -g)" \
        -e HOME=/tmp \
        -v "$PWD":/work -w /work \
        "$IMAGE" "$@"
}

LATEXMK=(latexmk -pdf -outdir=build -interaction=nonstopmode -file-line-error -halt-on-error)

case "${1:-build}" in
    build)
        mkdir -p build
        run_tex "${LATEXMK[@]}" main.tex
        cp build/main.pdf main.pdf
        echo "✅ Built main.pdf"
        ;;
    watch)
        mkdir -p build
        echo "Watching for changes (Ctrl+C to stop); PDF at build/main.pdf"
        run_tex "${LATEXMK[@]}" -pvc -view=none main.tex
        ;;
    clean)
        rm -rf build main.pdf
        echo "Cleaned build/ and main.pdf"
        ;;
    *)
        echo "Unknown command: $1 (expected build, watch, or clean)" >&2
        exit 1
        ;;
esac
