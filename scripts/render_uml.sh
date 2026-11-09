#!/usr/bin/env bash
# Render thesis/uml/*.puml to PNG and SVG in thesis/figures/.
# Needs PlantUML: brew install plantuml   (uses Java + Graphviz, both already installed)
set -euo pipefail
cd "$(dirname "$0")/.."
command -v plantuml >/dev/null || { echo "install PlantUML first: brew install plantuml"; exit 1; }
mkdir -p thesis/figures
for f in thesis/uml/*.puml; do
  plantuml -tpng -o "$PWD/thesis/figures" "$f"
  plantuml -tsvg -o "$PWD/thesis/figures" "$f"
done
ls thesis/figures/*.png | sed 's/^/  /'
