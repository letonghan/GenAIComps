@@ -0,0 +1,15 @@
#!/bin/bash

FILE="/home/user/.local/lib/python3.11/site-packages/langchain_milvus/vectorstores/milvus.py"

MARKER='if index is not None:'

TEMP_FILE=$(mktemp)

cat <<EOL > "$TEMP_FILE"
                index["index_param"] = self.index_params
EOL

sed -i "/$MARKER/r $TEMP_FILE" "$FILE"

rm "$TEMP_FILE"