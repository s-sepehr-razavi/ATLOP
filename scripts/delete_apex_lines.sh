#!/bin/bash

# Usage: ./delete_lines.sh your_script.py

FILE="$1"

if [ -z "$FILE" ]; then
  echo "Usage: $0 <filename.py>"
  exit 1
fi

if [ ! -f "$FILE" ]; then
  echo "File not found: $FILE"
  exit 1
fi

# Delete lines 36 to 49 using sed (in-place edit)
sed -i.bak '36,49d' "$FILE"

echo "Deleted lines 36 to 49 from $FILE (backup saved as $FILE.bak)"
