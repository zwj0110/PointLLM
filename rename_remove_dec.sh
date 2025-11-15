#!/usr/bin/env bash
# Usage: ./rename_remove_dec.sh /path/to/dir

DIR=${1:-.}

for f in "$DIR"/*_dec.npy; do
  # 如果没有匹配文件就跳过
  [ -e "$f" ] || continue
  # 去掉后缀 _dec.npy，改为 .npy
  mv -- "$f" "${f%_dec.npy}.npy"
  echo "Renamed: $(basename "$f") → $(basename "${f%_dec.npy}.npy")"
done
