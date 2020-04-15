#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = '80022068'
__mtime__ = '2019/8/19'
# qq:2456056533


"""
import logging.config
import os
from functools import wraps
from threading import Lock

cur_path = os.path.abspath(os.path.dirname((__file__)))
Log_Path = os.path.join(cur_path, 'logs')
if not os.path.exists(Log_Path):
    os.makedirs(Log_Path)


def singleton(cls):
    '''单例'''
    instances = {}
    locker = Lock()

    @wraps(cls)
    def _singleton(*args, **kwargs):
        if cls not in instances:
            with locker:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)

        return instances[cls]

    return _singleton


# @singleton
class KLogger:
    def __init__(self, log_path=Log_Path, info_name='info_k', errors_name='errors_k'):
        # size滚动
        size_handlers = {

            "console": {"class": "logging.StreamHandler", "level": "DEBUG", "formatter": "simple",
                        "stream": "ext://sys.stdout"},

            "info_file_handler": {"class": "logging.handlers.RotatingFileHandler", "level": "INFO",
                                  "formatter": "simple", "filename": "./logs/info.log", "maxBytes": 10485760,
                                  "backupCount": 8, "encoding": "utf8"},

            "error_file_handler": {"class": "logging.handlers.RotatingFileHandler", "level": "ERROR",
                                   "formatter": "simple", "filename": "./logs/errors.log", "maxBytes": 10485760,
                                   "backupCount": 8, "encoding": "utf8"}}

        # 时间滚动
        day_handlers = {"console": {"class": "logging.StreamHandler", "level": "DEBUG", "formatter": "simple",
                                    "stream": "ext://sys.stdout"},
                        "info_file_handler": {"class": "logging.handlers.TimedRotatingFileHandler", "level": "INFO",
                                              "formatter": "simple", "filename": "../log/info.log", "when": "midnight",
                                              "interval": 0, "backupCount": 8, "encoding": "utf8"},
                        "error_file_handler": {"class": "logging.handlers.TimedRotatingFileHandler", "level": "ERROR",
                                               "formatter": "simple", "filename": "../log/errors.log",
                                               "when": "midnight",
                                               "interval": 0, "backupCount": 8, "encoding": "utf8"}}

        _config = {"version": 1, "disable_existing_loggers": False,

                   "formatters": {"simple": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}},

                   "handlers": day_handlers,

                   "loggers": {"mymodule": {"level": "ERROR", "handlers": ["info_file_handler"], "propagate": "no"}},

                   "root": {"level": "INFO", "handlers": ["console", "info_file_handler", "error_file_handler"]}}

        _config['handlers']['info_file_handler']['filename'] = log_path + '/{}.log'.format(info_name)
        _config['handlers']['error_file_handler']['filename'] = log_path + '/{}.log'.format(errors_name)
        self.logger = logging.getLogger()
        logging.config.dictConfig(_config)


klogger = KLogger().logger


def klog(param):
    '''日志装饰器
        @klog: 默认模式
        @klog()： 可添加方法描述，param 可以是任意类型,方法不走 try
        @klog('excpt') : excpt 关键字，方法 try 处理

    '''
    if callable(param):
        def wrapper(*args, **kwargs):
            func_reslut = param(*args, **kwargs)
            klogger.info('===func_name: {},func_reslut:{}'.format(param.__name__, func_reslut))
            return func_reslut

        return wrapper

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if param == 'excpt':
                try:
                    func_reslut = func(*args, **kwargs)
                    klogger.info(
                        '===func_name:{},func_desc:{},func_reslut:{}'.format(func.__name__, param, func_reslut))
                    return func_reslut
                except Exception as e:
                    klogger.info('===func_name:{},func_error:{}'.format(func.__name__, e))
            else:
                func_reslut = func(*args, **kwargs)
                klogger.info('===func_name:{},func_desc:{},func_reslut:{}'.format(func.__name__, param, func_reslut))
                return func_reslut

        return wrapper

    return decorator


class KlogTest:

    def __init__(self, name, age):
        self.name = name
        self.age = age

    @klog
    def getName(self):
        return self.name

    @klog('只能拿到方法名称，获取不到类名')
    def getAge(self):
        return self.age

    @klog({'desc': '方法描述任意类型'})
    def getAge2(self):
        return self.age

    @klog(['方法描述任意类型'])
    def getAge3(self):
        return self.age

    @klog('excpt')
    def debug(self):
        return self.age / 0

    @klog('也可手动try')
    def debug2(self):
        try:
            return self.age / 0
        except Exception as e:
            klogger.info('====手动try error:{}'.format(e))
            pass
            # raise e


if __name__ == '__main__':
    a = KlogTest('jock', 2)
    print(a.getName())
    print(a.getAge())
    print(a.getAge2())
    print(a.getAge3())
    print(a.debug())
    print(a.debug2())
