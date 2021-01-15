""" dynamically load settings

author baiyu
"""
import conf.global_settings as settings
import conf.global_settings_cv as settings_cv

class Settings:
    def __init__(self, settings):

        for attr in dir(settings):
            if attr.isupper():
                setattr(self, attr, getattr(settings, attr))

settings = Settings(settings)
settings_cv = Settings(settings_cv)