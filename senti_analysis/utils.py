# encoding: utf8

import json
import jieba


def dump(json_obj, file_path):
    """
    Serializer json object.
    :param json_obj: json object
    :param file_path: file path
    :return: None
    """
    with open(file_path, 'w+') as f:
        json.dump(json_obj, f, ensure_ascii=False)


def load(file_path):
    """
    Deserializer object
    :param file_path: file path
    :return:
    """
    with open(file_path) as f:
        return json.load(f)


def cut(text):
    return jieba.lcut(text)
