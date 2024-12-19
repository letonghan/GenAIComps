# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import urllib.parse


def encode_filename(filename):
    return urllib.parse.quote(filename, safe="")


def decode_filename(encoded_filename):
    return urllib.parse.unquote(encoded_filename)


def format_search_results(response, file_list: list):
    for i in range(1, len(response), 2):
        file_name = response[i].decode()[5:]
        file_dict = {
            "name": decode_filename(file_name),
            "id": decode_filename(file_name),
            "type": "File",
            "parent": "",
        }
        file_list.append(file_dict)
    return file_list
