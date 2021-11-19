from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.core.window import Window

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.utils import platform
import time
import os
import subprocess
from kivy.properties import (ObjectProperty, StringProperty, BooleanProperty, ListProperty)

import _init_paths
from lib import mycamera

# Create the screen manager
Builder.load_file("mycamera.kv")

sm = ScreenManager()
sm.add_widget(mycamera.mycamera(name='mycamera_screen'))


class cameraApp(App):
    def build(self):
        return sm

if __name__ == '__main__':
    cameraApp().run()
