#!/bin/bash

FILE="/home/user/.local/lib/python3.11/site-packages/langchain_milvus/vectorstores/milvus.py"

NEW_LINE='\\t\t\t\tindex["index_param"] = self.index_params\n'

MARKER='if index is not None:'

sed -i "/$MARKER/a $NEW_LINE" "$FILE"
