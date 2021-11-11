from kivy.core.window import Window
from kivy.uix.screenmanager import Screen
from kivy.event import EventDispatcher

from kivy.utils import platform

import numpy as np
import os
from jnius import autoclass, cast
from time import sleep

from android.permissions import request_permissions, check_permission, Permission
request_permissions([
    Permission.RECORD_AUDIO,
    Permission.CAMERA,
    Permission.WRITE_EXTERNAL_STORAGE,
    Permission.READ_EXTERNAL_STORAGE
])


from android.storage import primary_external_storage_path
SD_CARD = primary_external_storage_path()
print('SDCARD is on :')
print(SD_CARD)


if platform == 'android':
    print('The platform is Android')

    # get the needed Java classes
    MediaRecorder = autoclass('android.media.MediaRecorder')

    AudioSource = autoclass('android.media.MediaRecorder$AudioSource')
    AudioEncoder = autoclass('android.media.MediaRecorder$AudioEncoder')

    VideoSource = autoclass('android.media.MediaRecorder$VideoSource')
    VideoEncoder = autoclass('android.media.MediaRecorder$VideoEncoder')

    OutputFormat = autoclass('android.media.MediaRecorder$OutputFormat')



from kivy.graphics.texture import Texture
from kivy.graphics import Fbo, Callback, Rectangle
from kivy.properties import (BooleanProperty, StringProperty, ObjectProperty, OptionProperty, ListProperty)
from kivy.clock import Clock

from jnius import autoclass, cast, PythonJavaClass, java_method, JavaClass, MetaJavaClass, JavaMethod

import logging
from functools import partial
from enum import Enum
from kivy.animation import Animation


logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

CameraManager = autoclass("android.hardware.camera2.CameraManager")
PythonActivity = autoclass("org.kivy.android.PythonActivity")
Context = autoclass("android.content.Context")
context = cast("android.content.Context", PythonActivity.mActivity)

CameraDevice = autoclass("android.hardware.camera2.CameraDevice")
CaptureRequest = autoclass("android.hardware.camera2.CaptureRequest")
CameraCharacteristics = autoclass("android.hardware.camera2.CameraCharacteristics")

ArrayList = autoclass('java.util.ArrayList')
JavaArray = autoclass('java.lang.reflect.Array')

SurfaceTexture = autoclass('android.graphics.SurfaceTexture')
Surface = autoclass('android.view.Surface')
GL_TEXTURE_EXTERNAL_OES = autoclass(
    'android.opengl.GLES11Ext').GL_TEXTURE_EXTERNAL_OES
ImageFormat = autoclass('android.graphics.ImageFormat')

Handler = autoclass("android.os.Handler")
Looper = autoclass("android.os.Looper")

MyStateCallback = autoclass("mycamera2.MyStateCallback")
CameraActions = autoclass("mycamera2.MyStateCallback$CameraActions")
# MyStateCallback = autoclass("org.kivy.android.MyStateCallback")

MyCaptureSessionCallback = autoclass("mycamera2.MyCaptureSessionCallback")
CameraCaptureEvents = autoclass("mycamera2.MyCaptureSessionCallback$CameraCaptureEvents")

_global_handler = Handler(Looper.getMainLooper())

class LensFacing(Enum):
    """Values copied from CameraCharacteristics api doc, as pyjnius
    lookup doesn't work on some devices.
    """
    LENS_FACING_FRONT = 0
    LENS_FACING_BACK = 1
    LENS_FACING_EXTERNAL = 2

class ControlAfMode(Enum):
    CONTROL_AF_MODE_CONTINUOUS_PICTURE = 4

class ControlAeMode(Enum):
    CONTROL_AE_MODE_ON = 1

class Runnable(PythonJavaClass):
    __javainterfaces__ = ['java/lang/Runnable']

    def __init__(self, func):
        super(Runnable, self).__init__()
        self.func = func

    @java_method('()V')
    def run(self):
        try:
            self.func()
        except:
            import traceback
            traceback.print_exc()

class PyCameraInterface(EventDispatcher):
    """
    Provides an API for querying details of the cameras available on Android.
    """

    camera_ids = []

    cameras = ListProperty()

    java_camera_characteristics = {}

    java_camera_manager = ObjectProperty()

    def __init__(self):
        super().__init__()
        logger.info("Starting camera interface init")
        self.java_camera_manager = cast("android.hardware.camera2.CameraManager",
                                    context.getSystemService(Context.CAMERA_SERVICE))

        self.camera_ids = self.java_camera_manager.getCameraIdList()
        characteristics_dict = self.java_camera_characteristics
        camera_manager = self.java_camera_manager
        logger.info("Got basic java objects")
        for camera_id in self.camera_ids:
            logger.info(f"Getting data for camera {camera_id}")
            characteristics_dict[camera_id] = camera_manager.getCameraCharacteristics(camera_id)
            logger.info("Got characteristics dict")

            self.cameras.append(PyCameraDevice(
                camera_id=camera_id,
                java_camera_manager=camera_manager,
                java_camera_characteristics=characteristics_dict[camera_id],
            ))
            logger.info(f"Finished interpreting camera {camera_id}")

    def select_cameras(self, **conditions):
        options = self.cameras
        outputs = []
        for camera in cameras:
            for key, value in conditions.items():
                if getattr(camera, key) != value:
                    break
            else:
                outputs.append(camera)

        return outputs

class PyCameraDevice(EventDispatcher):

    camera_id = StringProperty()

    output_texture = ObjectProperty(None, allownone=True)

    preview_active = BooleanProperty(False)
    preview_texture = ObjectProperty(None, allownone=True)
    preview_resolution = ListProperty()
    preview_fbo = ObjectProperty(None, allownone=True)
    java_preview_surface_texture = ObjectProperty(None)
    java_preview_surface = ObjectProperty(None)
    ##
    java_recorder_surface = ObjectProperty(None)
    ##
    java_capture_request = ObjectProperty(None)
    java_surface_list = ObjectProperty(None)
    java_capture_session = ObjectProperty(None)

    connected = BooleanProperty(False)

    supported_resolutions = ListProperty()
    # TODO: populate this

    facing = OptionProperty("UNKNOWN", options=["UNKNOWN", "FRONT", "BACK", "EXTERNAL"])

    java_camera_characteristics = ObjectProperty()
    java_camera_manager = ObjectProperty()
    java_camera_device = ObjectProperty()
    java_stream_configuration_map = ObjectProperty()

    _open_callback = ObjectProperty(None, allownone=True)


    ##
    bIsRecording = BooleanProperty(False)
    mRecorder = MediaRecorder()
    ##

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_event_type("on_opened")
        self.register_event_type("on_closed")
        self.register_event_type("on_disconnected")
        self.register_event_type("on_error")

        self._java_state_callback_runnable = Runnable(self._java_state_callback)
        self._java_state_java_callback = MyStateCallback(self._java_state_callback_runnable)

        self._java_capture_session_callback_runnable = Runnable(self._java_capture_session_callback)
        self._java_capture_session_java_callback = MyCaptureSessionCallback(
            self._java_capture_session_callback_runnable)

        self._populate_camera_characteristics()


        #####
        #####

    def on_opened(self, instance):
        pass
    def on_closed(self, instance):
        pass
    def on_disconnected(self, instance):
        pass
    def on_error(self, instance, error):
        pass

    def close(self):
        self.java_camera_device.close()

    def _populate_camera_characteristics(self):
        logger.info("Populating camera characteristics")
        self.java_stream_configuration_map = self.java_camera_characteristics.get(
            CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
        logger.info("Got stream configuration map")

        self.supported_resolutions = [
            (size.getWidth(), size.getHeight()) for size in
            self.java_stream_configuration_map.getOutputSizes(SurfaceTexture(0).getClass())]
        logger.info("Got supported resolutions")

        facing = self.java_camera_characteristics.get(
            CameraCharacteristics.LENS_FACING)
        logger.info(f"Got facing: {facing}")
        if facing == LensFacing.LENS_FACING_BACK.value:  # CameraCharacteristics.LENS_FACING_BACK:
            self.facing = "BACK"
        elif facing == LensFacing.LENS_FACING_FRONT.value:  # CameraCharacteristics.LENS_FACING_FRONT:
            self.facing = "FRONT"
        elif facing == LensFacing.LENS_FACING_EXTERNAL.value:  # CameraCharacteristics.LENS_FACING_EXTERNAL:
            self.facing = "EXTERNAL"
        else:
            raise ValueError("Camera id {} LENS_FACING is unknown value {}".format(self.camera_id, facing))
        logger.info(f"Finished initing camera {self.camera_id}")

    def __str__(self):
        return "<PyCameraDevice facing={}>".format(self.facing)
    def __repr__(self):
        return str(self)

    def open(self, callback=None):
        self._open_callback = callback
        self.java_camera_manager.openCamera(
            self.camera_id,
            self._java_state_java_callback,
            _global_handler
        )

    def _java_state_callback(self, *args, **kwargs):
        action = MyStateCallback.camera_action.toString()
        camera_device = MyStateCallback.camera_device

        self.java_camera_device = camera_device

        logger.info("CALLBACK: camera event {}".format(action))
        if action == "OPENED":
            self.dispatch("on_opened", self)
            self.connected = True
        elif action == "DISCONNECTED":
            self.dispatch("on_disconnected", self)
            self.connected = False
        elif action == "CLOSED":
            self.dispatch("on_closed", self)
            self.connected = False
        elif action == "ERROR":
            error = MyStateCallback.camera_error
            self.dispatch("on_error", self, error)
            self.connected = False
        elif action == "UNKNOWN":
            print("UNKNOWN camera state callback item")
            self.connected = False
        else:
            raise ValueError("Received unknown camera action {}".format(action))

        if self._open_callback is not None:
            self._open_callback(self, action)

    def start_preview(self, resolution, fps, video_out_path):
        if self.java_camera_device is None:
            raise ValueError("Camera device not yet opened, cannot create preview stream")
        
        if resolution not in self.supported_resolutions:
            raise ValueError(
                "Tried to open preview with resolution {}, not in supported resolutions {}".format(
                    resolution, self.supported_resolutions))
        
        if self.preview_active:
            raise ValueError("Preview already active, can't start again without stopping first")

        logger.info("Creating capture stream with resolution {}".format(resolution))

        self.preview_resolution = resolution
        self._prepare_preview_fbo(resolution)
        self.preview_texture = Texture(
            width=resolution[0], height=resolution[1], target=GL_TEXTURE_EXTERNAL_OES, colorfmt="rgba")
        logger.info("Texture id is {}".format(self.preview_texture.id))
        self.java_preview_surface_texture = SurfaceTexture(int(self.preview_texture.id))
        self.java_preview_surface_texture.setDefaultBufferSize(*resolution)
        self.java_preview_surface = Surface(self.java_preview_surface_texture)

        
        ##################
        self.mRecorder.setVideoSource(VideoSource.SURFACE)
        self.mRecorder.setOutputFormat(OutputFormat.MPEG_4)

        if not os.path.exists(os.path.join(SD_CARD,video_out_path)):
            open(os.path.join(SD_CARD,video_out_path), 'w').close()
        self.mRecorder.setOutputFile(os.path.join(SD_CARD,video_out_path))

        self.mRecorder.setVideoEncodingBitRate(1600 * 1000)
        self.mRecorder.setVideoFrameRate(fps)

        self.mRecorder.setVideoSize(resolution[0],resolution[1])
        self.mRecorder.setVideoEncoder(VideoEncoder.H264)
        # self.current_camera.mRecorder.setMaxDuration(-1)
        logger.info("Preview is set")
        try:
            self.mRecorder.prepare()        
        except:
            print('Video recording preparation error')


        self.java_recorder_surface = self.mRecorder.getSurface()
        



        self.java_capture_request = self.java_camera_device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
        self.java_capture_request.addTarget(self.java_preview_surface)
        
        
        ##
        self.java_capture_request.addTarget(self.java_recorder_surface)
        ##
        

        self.java_capture_request.set(
            CaptureRequest.CONTROL_AF_MODE, ControlAfMode.CONTROL_AF_MODE_CONTINUOUS_PICTURE.value)  # CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
        self.java_capture_request.set(
            CaptureRequest.CONTROL_AE_MODE, ControlAeMode.CONTROL_AE_MODE_ON.value)  # CaptureRequest.CONTROL_AE_MODE_ON)











        self.java_surface_list = ArrayList()
        self.java_surface_list.add(self.java_preview_surface)

        
        ##
        self.java_surface_list.add(self.java_recorder_surface)
        ##
        

        self.java_camera_device.createCaptureSession(
            self.java_surface_list,
            self._java_capture_session_java_callback,
            _global_handler,
        )

        return self.preview_fbo.texture



    def start_preview_record(self, resolution, fps, video_path):
        if self.java_camera_device is None:
            raise ValueError("Camera device not yet opened, cannot create preview stream")
        
        if resolution not in self.supported_resolutions:
            raise ValueError(
                "Tried to open preview with resolution {}, not in supported resolutions {}".format(
                    resolution, self.supported_resolutions))
        
        if self.preview_active:
            raise ValueError("Preview already active, can't start again without stopping first")

        logger.info("Creating recording stream with resolution {}".format(resolution))

        ##################
        self.mRecorder.setVideoSource(VideoSource.SURFACE)
        self.mRecorder.setOutputFormat(OutputFormat.MPEG_4)

        if not os.path.exists(os.path.join(SD_CARD,video_path)):
            open(os.path.join(SD_CARD,video_path), 'w').close()
        self.mRecorder.setOutputFile(os.path.join(SD_CARD,video_path))

        self.mRecorder.setVideoEncodingBitRate(1600 * 1000)
        self.mRecorder.setVideoFrameRate(fps)

        self.mRecorder.setVideoSize(resolution[0],resolution[1])
        self.mRecorder.setVideoEncoder(VideoEncoder.H264)
        # self.current_camera.mRecorder.setMaxDuration(-1)
        logger.info("Recording is set")
        try:
            self.mRecorder.prepare()        
        except:
            print('Video recording preparation error')


        self.java_recorder_surface = self.mRecorder.getSurface()
        



        self.java_capture_request = self.java_camera_device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
        self.java_capture_request.addTarget(self.java_recorder_surface)
        ##
        
        self.java_capture_request.set(
            CaptureRequest.CONTROL_AF_MODE, ControlAfMode.CONTROL_AF_MODE_CONTINUOUS_PICTURE.value)  # CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
        self.java_capture_request.set(
            CaptureRequest.CONTROL_AE_MODE, ControlAeMode.CONTROL_AE_MODE_ON.value)  # CaptureRequest.CONTROL_AE_MODE_ON)



        self.java_surface_list = ArrayList()
    
        ##
        self.java_surface_list.add(self.java_recorder_surface)
        ##

        self.java_camera_device.createCaptureSession(
            self.java_surface_list,
            self._java_capture_session_java_callback,
            _global_handler,
        )

        return None



    def _prepare_preview_fbo(self, resolution):
        self.preview_fbo = Fbo(size=resolution)
        self.preview_fbo['resolution'] = [float(f) for f in resolution]
        self.preview_fbo.shader.fs = """
            #extension GL_OES_EGL_image_external : require
            #ifdef GL_ES
                precision highp float;
            #endif

            /* Outputs from the vertex shader */
            varying vec4 frag_color;
            varying vec2 tex_coord0;

            /* uniform texture samplers */
            uniform sampler2D texture0;
            uniform samplerExternalOES texture1;
            uniform vec2 resolution;

            void main()
            {
                gl_FragColor = texture2D(texture1, tex_coord0);
            }
        """
        with self.preview_fbo:
            Rectangle(size=resolution)

    def _java_capture_session_callback(self, *args, **kwargs):
        event = MyCaptureSessionCallback.camera_capture_event.toString()
        logger.info("CALLBACK: capture event {}".format(event))

        self.java_capture_session = MyCaptureSessionCallback.camera_capture_session

        if event == "READY":
            logger.info("Doing READY actions")
            self.java_capture_session.setRepeatingRequest(self.java_capture_request.build(), None, None)
            Clock.schedule_interval(self._update_preview, 0.)

    def _update_preview(self, dt):
        self.java_preview_surface_texture.updateTexImage()
        self.preview_fbo.ask_update()
        self.preview_fbo.draw()
        self.output_texture = self.preview_fbo.texture





from kivy.uix.widget import Widget
from kivy.uix.stencilview import StencilView


class CameraDisplayWidget(StencilView):
    texture = ObjectProperty(None, allownone=True)

    resolution = ListProperty([1, 1])

    tex_coords = ListProperty([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    correct_camera = BooleanProperty(False)

    _rect_pos = ListProperty([0, 0])
    _rect_size = ListProperty([1, 1])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.bind(
            pos=self._update_rect,
            size=self._update_rect,
            resolution=self._update_rect,
            texture=self._update_rect,
        )

    def on_correct_camera(self, instance, correct):
        print("Correct became", correct)
        if correct:
            self.tex_coords = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
            print("Set 0!")
        else:
            self.tex_coords = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
            print("Set 1!")

    def on_tex_coords(self, instance, value):
        print("tex_coords became", self.tex_coords)

    def _update_rect(self, *args):
        self._update_rect_to_fill()

    def _update_rect_to_fit(self, *args):
        w, h = self.resolution
        aspect_ratio = h / w

        aspect_width = self.width
        aspect_height = self.width * h / w
        if aspect_height > self.height:
            aspect_height = self.height
            aspect_width = aspect_height * w / h

        aspect_height = int(aspect_height)
        aspect_width = int(aspect_width)

        self._rect_pos = [self.center_x - aspect_width / 2,
                          self.center_y - aspect_height / 2]

        self._rect_size = [aspect_width, aspect_height]

    def _update_rect_to_fill(self, *args):
        w, h = self.resolution

        aspect_ratio = h / w

        aspect_width = self.width
        aspect_height = self.width * h / w
        if aspect_height < self.height:
            aspect_height = self.height
            aspect_width = aspect_height * w / h

        aspect_height = int(aspect_height)
        aspect_width = int(aspect_width)

        self._rect_pos = [self.center_x - aspect_width / 2,
                          self.center_y - aspect_height / 2]

        self._rect_size = [aspect_width, aspect_height]


class PermissionRequestStates(Enum):
    UNKNOWN = "UNKNOWN"
    HAVE_PERMISSION = "HAVE_PERMISSION"
    DO_NOT_HAVE_PERMISSION = "DO_NOT_HAVE_PERMISSION"
    AWAITING_REQUEST_RESPONSE = "AWAITING_REQUEST_RESPONSE"


class mycamera(Screen):
    texture = ObjectProperty(None, allownone=True)
    # camera_resolution = ListProperty([1024, 768])
    current_camera = ObjectProperty(None, allownone=True)
    cameras_to_use = ListProperty()

    ###########################################
    camera_permission_state = OptionProperty(
        PermissionRequestStates.UNKNOWN,
        options=[PermissionRequestStates.UNKNOWN,
                 PermissionRequestStates.HAVE_PERMISSION,
                 PermissionRequestStates.DO_NOT_HAVE_PERMISSION,
                 PermissionRequestStates.AWAITING_REQUEST_RESPONSE])
    _camera_permission_state_string = StringProperty("UNKNOWN")
    ###############################
    

    def __init__(self, **kwargs):
        self.fps = 15
        self.video_out_path = 'output.mp4'
        self.video_resolution = (1024,768)
        self.texture_resolution = (1024,768)
        super().__init__(**kwargs)


    def on_camera_permission_state(self, instance, state):
        self._camera_permission_state_string = state.value






    def start_camera(self):
        print('start_camera is called')
        self.camera_interface = PyCameraInterface()
        self.camera_update = Clock.schedule_interval(self.update, 0)
        self.debug_print_camera_info()
        self.inspect_cameras()
        self.restart_stream()
        
        
        
        
        




        

    def change_set_camera(self):
        print('change_camera is called')
        self.ensure_camera_closed()
        self.cameras_to_use = self.cameras_to_use[1:] + [self.cameras_to_use[0]]

        self.cameras_to_use[0].mRecorder = None
        self.cameras_to_use[0].mRecorder = MediaRecorder()
        
        self.attempt_stream_camera(self.cameras_to_use[0])
        
    def capture_video(self):
        print('Capture comand is called')
        #self.current_camera.start_preview(tuple(self.texture_resolution), self.fps, self.video_out_path)
        
        self.current_camera.mRecorder.start()

    def save_video(self, *_):
        print('video is saved')
        self.current_camera.mRecorder.stop()
        self.current_camera.mRecorder.release()
        
    def save_record_video(self, *_):
        print('video is saved and new recording session starts')
        self.current_camera.mRecorder.stop()   
        self.current_camera.mRecorder.release()
        self.ensure_camera_closed()

        self.cameras_to_use[0].mRecorder = None
        self.cameras_to_use[0].mRecorder = MediaRecorder()
        

        #self.current_camera.mRecorder = None
        #self.current_camera.mRecorder = MediaRecorder()
        
        self.video_out_path = 'output_2.mp4'        
                
        self.attempt_stream_camera(self.cameras_to_use[0])

        #self.attempt_stream_camera(self.current_camera)

        #self.current_camera.start_preview(tuple(self.texture_resolution), self.fps, self.video_out_path)
        
        #self.current_camera.mRecorder = None
        #self.current_camera.mRecorder = MediaRecorder()
        #self.attempt_stream_camera(self.current_camera)
        #self.restart_stream()
        #self.current_camera.mRecorder.start()
        self.clock_starter = Clock.schedule_interval(self.attempt_start_recorder,0.1)



    def pause_video(self, *_):
        print('video is saved')
        self.current_camera.mRecorder.pause()        
    def resume_video(self, *_):
        print('video is saved')
        self.current_camera.mRecorder.resume()        
        

    def attempt_start_recorder(self, *_):
        if self.current_camera:
            self.current_camera.mRecorder.start()
            Clock.unschedule(self.clock_starter)
            self.clock_starter = None

        







    def inspect_cameras(self):
        cameras = self.camera_interface.cameras

        for camera in cameras:
            if camera.facing == "BACK":
                self.cameras_to_use.append(camera)
        for camera in cameras:
            if camera.facing == "FRONT":
                self.cameras_to_use.append(camera)

    def restart_stream(self):
        self.ensure_camera_closed()
        Clock.schedule_once(self._restart_stream, 0)

    def _restart_stream(self, dt):
        logger.info("On restart, state is {}".format(self.camera_permission_state))
        if self.camera_permission_state in (PermissionRequestStates.UNKNOWN, PermissionRequestStates.HAVE_PERMISSION):
            self.attempt_stream_camera(self.cameras_to_use[0])
        else:
            logger.warning(
                "Did not attempt to restart camera stream as state is {}".format(self.camera_permission_state))

    def debug_print_camera_info(self):
        cameras = self.camera_interface.cameras
        camera_infos = ["Camera ID {}, facing {}".format(c.camera_id, c.facing) for c in cameras]
        for camera in cameras:
            print("Camera ID {}, facing {}, resolutions {}".format(
                camera.camera_id, camera.facing, camera.supported_resolutions))

    def stream_camera_index(self, index):
        self.attempt_stream_camera(self.camera_interface.cameras[index])

    def attempt_stream_camera(self, camera):
        """Start streaming from the given camera, if we have the CAMERA
        permission, otherwise request the permission first.
        """

        if check_permission(Permission.CAMERA):
            self.stream_camera(camera)
        else:
            self.camera_permission_state = PermissionRequestStates.AWAITING_REQUEST_RESPONSE
            request_permission(Permission.CAMERA, partial(self._request_permission_callback, camera))

    def _request_permission_callback(self, camera, permissions, alloweds):
        # Assume  that we  received info  about exactly  1 permission,
        # since we only ever ask for CAMERA
        allowed = alloweds[0]

        if allowed:
            self.camera_permission_state = PermissionRequestStates.HAVE_PERMISSION
            self.stream_camera(camera)
        else:
            self.camera_permission_state = PermissionRequestStates.DO_NOT_HAVE_PERMISSION
            print("PERMISSION FORBIDDEN")

    def stream_camera(self, camera):
        resolution = self.select_texture_resolution(Window.size, camera.supported_resolutions)
        if resolution is None:
            logger.error(f"Found no good resolution in {camera.supported_resolutions} for Window.size {Window.size}")
            return
        else:
            logger.info(f"Chose resolution {resolution} from choices {camera.supported_resolutions}")
        self.texture_resolution = resolution
        camera.open(callback=self._stream_camera_open_callback)

    def _stream_camera_open_callback(self, camera, action):
        if action == "OPENED":
            logger.info("Camera opened, preparing to start preview")
            Clock.schedule_once(partial(self._stream_camera_start_preview, camera), 0)
        else:
            logger.info("Ignoring camera event {action}")

    def _stream_camera_start_preview(self, camera, *args):
        logger.info("Starting preview of camera {camera}")
        if camera.facing == "FRONT":
            self.ids.cdw.correct_camera = True
        else:
            self.ids.cdw.correct_camera = False
        self.texture = camera.start_preview(tuple(self.texture_resolution), self.fps, self.video_out_path)

        self.current_camera = camera
    '''
    def select_resolution(self, window_size, resolutions, best=None):
        if best in resolutions:
            return best

        if not resolutions:
            return None

        win_x, win_y = window_size
        larger_resolutions = [(x, y) for (x, y) in resolutions if (x > win_x and y > win_y)]


        if larger_resolutions:
            return min(larger_resolutions, key=lambda r: r[0] * r[1])

        smaller_resolutions = resolutions  # if we didn't find one yet, all are smaller than the requested Window size
        return max(smaller_resolutions, key=lambda r: r[0] * r[1])
    '''
    def select_texture_resolution(self, window_size, resolutions):
        if not resolutions:
            return None
        
        smaller_resolutions = [(x, y) for (x, y) in resolutions if (x < self.ids.cdw.size[0] and y < self.ids.cdw.size[1])]
        print('My smaller resolutions:')
        print(smaller_resolutions)
        if smaller_resolutions:
            return max(smaller_resolutions, key=lambda r: r[0] * r[1])
        else:
            return min(resolutions, key=lambda r: r[0] * r[1])

    def on_texture(self, instance, value):
        print("App texture changed to {}".format(value))

    def update(self, dt):
        self.canvas.ask_update()


    def ensure_camera_closed(self):
        if self.current_camera is not None:
            self.current_camera.close()
            self.current_camera = None

    def on_pause(self):

        logger.info("Closing camera due to pause")
        self.ensure_camera_closed()

        return super().on_pause()

    def on_resume(self):
        logger.info("Opening camera due to resume")
        self.restart_stream()




'''
    def capture_audio(self):
        print('Capture comand is called')

        self.mRecorder.setAudioSource(AudioSource.MIC)
        self.mRecorder.setOutputFormat(OutputFormat.THREE_GPP)
        self.mRecorder.setOutputFile(os.path.join(SD_CARD,'myaudio.3gp'))
        self.mRecorder.setAudioEncoder(AudioEncoder.AMR_NB)
        self.mRecorder.prepare()

        # record 5 seconds
        self.mRecorder.start()
        sleep(5)
        self.mRecorder.stop()
        self.mRecorder.release()

    def capture_video(self):
        print('Capture comand is called')
        
        self.camera_interface = PyCameraInterface()
        cameras = self.camera_interface.cameras

        for camera in cameras:
            if camera.facing == "BACK":
                self.cameras_to_use.append(camera)
        for camera in cameras:
            if camera.facing == "FRONT":
                self.cameras_to_use.append(camera)
        
        self.current_camera = self.cameras_to_use[0]

        

        self.mRecorder.setVideoSource(0)
        self.mRecorder.setOutputFormat(OutputFormat.MPEG_4)
        self.mRecorder.setOutputFile(os.path.join(SD_CARD,'myvideo1.avi'))
        self.mRecorder.setVideoEncoder(VideoEncoder.H264)
        self.mRecorder.setCaptureRate(10.0)

        self.mRecorder.prepare()


        self.current_camera.open(callback=self.stream_video)

    def stream_video(self, *_):
        print('streaming started')
        self.mRecorder.start()
        sleep(5)

    def save_video(self, *_):
        print('video is saved')
                # record 5 seconds
        self.mRecorder.stop()
        self.mRecorder.release()

'''
        
