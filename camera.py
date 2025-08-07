#!/usr/bin/env python3

from math import floor
import numpy as np
from json import load

import echolib
import cv2

from vimba import Vimba, FrameStatus, Camera, PixelFormat
import gc

from auto_calibration import ImageCorrection
from echolib.camera import FramePublisher, Frame

def setup_software_triggering(cam: Camera):
    # Always set the selector first so that folling features are applied correctly!
    cam.TriggerSelector.set('FrameStart')

    # optional in this example but good practice as it might be needed for hadware triggering
    cam.TriggerActivation.set('RisingEdge')

    # Make camera listen to Software trigger
    cam.TriggerSource.set('Software')
    cam.TriggerMode.set('On')

class VimbaCameraHandler():

    def __init__(self, image_correction):

        self.message = None
        self.frame = None
        self.n_frames = 0
        self.settings_changed = False
        
        self.image_correction = image_correction

        self.camera: Camera = None

    def frame_handler(self, camera, frame):
        if frame.get_status() == FrameStatus.Complete:
            frame_copy = frame.as_numpy_ndarray()
            height, width, _ = frame_copy.shape
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BAYER_RGGB2RGB)
            frame_copy = self.image_correction.correct(frame_copy)
            frame_copy = cv2.resize(frame_copy, (width//2, height//2))
            
            self.frame = np.array(frame_copy)

            self.n_frames += 1

        if self.settings_changed:
            self.settings_changed = False
            return
        camera.queue_frame(frame)

    def callback_camera_input(self, message):

        command, value = echolib.MessageReader(message).readString().split(" ")

        try:
            print(f"Got command: {command} -> {value}")

            if command == "ExposureAuto":
                self.camera.ExposureAuto.set(value)
            elif command == "BalanceWhiteAuto":
                self.camera.BalanceWhiteAuto.set(value)   
            elif command == "BalanceRatio":
                self.camera.BalanceRatio.set(float(value))
            elif command == "ExposureTime":
                self.camera.ExposureTime.set(float(value))
            elif command == "GetBalanceRatio":
                self.message = f"BalanceRatio {self.camera.BalanceRatio.get()}"
                return
            elif command == "GetExposureTime":
                self.message = f"ExposureTime {self.camera.ExposureTime.get()}"
                return
            
            self.camera.stop_streaming()
            self.settings_changed = True
            gc.collect() # free the stream buffer

            self.camera.AcquisitionFrameRateEnable.set(True)
            range = self.camera.get_feature_by_name("AcquisitionFrameRate").get_range()
            fr = floor(range[1])

            print(f"Frame rate range: [{range[0]}, {range[1]}]: setting {fr}")

            try:
                self.camera.AcquisitionFrameRate.set(fr)
            except Exception as e:
                    print(f"[ERROR] Setting camera frame rate failed: {e}") 

            self.camera.start_streaming(self.frame_handler, buffer_count = 100)
        except Exception as e:
            print(f"[ERROR] Error while setting {command} = {value} | error print -> {e}")
            

def setup_args():
    import argparse
    parser = argparse.ArgumentParser(prog = 'Vimba Camera Streamer')
    parser.add_argument('--config', default='/opt/config/camera0.json', help='Path to camera configuration json file')  
    
    args = parser.parse_args()
    
    return args

def load_correction_data(config):
    ref_cal_path   = config.get("ReferenceCalPath")
    ref_white_path = config.get("ReferenceWhitePath")
    
    illum_path = config.get("IlluminationMapPath")
    ccm_path = config.get("ColorCorrectionMatrixPath")
    
    illum_map = np.load(illum_path).astype(np.float32)
    ccm = np.load(ccm_path).astype(np.float32)

    illumination_settings = {
        "smoothing_sigma": config.get("IlluminationSmoothingSigma", 301),
        "sample_step": config.get("IlluminationSampleStep", 50),
        "region_size": config.get("IlluminationRegionSize", 10),
        "interpolation_method": config.get("IlluminationInterpolationMethod", "cubic"),
        "visualize": False
    }

    color_settings = {
        "grid_shape": (config.get("ColorCorrectionGridRows", 4), config.get("ColorCorrectionGridColumns", 7)),
        "sample_size": config.get("ColorCorrectionSampleSize", 50),
        "visualize": False
    }

    reference_paths = {
        "cal": ref_cal_path,
        "white": ref_white_path,
    }
    
    return reference_paths, illum_map, ccm, illumination_settings, color_settings

def main():
    args = setup_args()
    
    ###########################
    # Load camera parameters  #
    ###########################

    def cameraSetBalanceRatioOnChannel(cam,v,ch):
        cam.BalanceRatioSelector.set(ch)
        cam.BalanceRatio.set(v)

    config = load(open(args.config))
    config_set_functions = \
    {
        "ExposureAuto": lambda camera, v: camera.ExposureAuto.set(v),
        "ExposureTime": lambda camera, v: camera.ExposureTime.set(v),
        "BalanceWhiteAuto": lambda camera, v: camera.BalanceWhiteAuto.set(v),
        "BalanceRatio":     lambda camera, v: camera.BalanceRatio.set(v),
        "BalanceRatioRed":     lambda camera, v: cameraSetBalanceRatioOnChannel(camera,v,"Red"),
        "BalanceRatioBlue":     lambda camera, v: cameraSetBalanceRatioOnChannel(camera,v,"Blue"),
        "DeviceLinkThroughputLimit":  lambda camera, v: camera.DeviceLinkThroughputLimit.set(int(v)),
        "AcquisitionFrameRateEnable": lambda camera, v: camera.AcquisitionFrameRateEnable.set(bool(int(v))),
    }
    
    ###############################
    # Load correction parameters  #
    ###############################
    
    reference_paths, illum_map, ccm, illumination_settings, color_settings = load_correction_data(config=config)
    image_correction = ImageCorrection(
        reference_paths=reference_paths, 
        illumination_settings=illumination_settings, 
        color_settings=color_settings,
        illum_map=illum_map,
        ccm=ccm
    )

    ###########################

    handler = VimbaCameraHandler(image_correction=image_correction)

    loop   = echolib.IOLoop()
    client = echolib.Client()
    loop.add_handler(client)

    output = FramePublisher(client, "camera_stream_0")
    command_input  = echolib.Subscriber(client, "camera_stream_0_input", "string", handler.callback_camera_input)
    command_output = echolib.Publisher(client, "camera_stream_0_output", "string")

    loop.wait(100)
    
    ###########################

    with Vimba.get_instance() as vimba_instance:

        vimba_cameras = vimba_instance.get_all_cameras()

        with vimba_cameras[0] as vimba_camera:

            handler.camera = vimba_camera
    
            ###########################
            # Grab relevant ranges
            ###########################

            for f in ["BalanceRatio", "ExposureTime"]:
                range = vimba_camera.get_feature_by_name(f).get_range()

                writer = echolib.MessageWriter()
                writer.writeString(f"{f}Range {range[0]} {range[1]}")

                command_output.send(writer)

            ###########################
            # Set default camera values
            ###########################

            if vimba_camera.get_pixel_format() != PixelFormat.BayerRG8:
                print("Setting pixel format to", PixelFormat.BayerRG8, "(hardcoded)")
                vimba_camera.set_pixel_format(PixelFormat.BayerRG8)
            for feature in config_set_functions:
                if feature in config:
                    try:
                        print(f"Setting default feature {feature} with {config[feature]}")

                        config_set_functions[feature](vimba_camera, config[feature])
                    except Exception as e:
                        print(f"[ERROR] Setting camera feature {feature} failer: {e}")

            
            # Send default values to ui
            writer = echolib.MessageWriter()
            writer.writeString(f"BalanceRatio {vimba_camera.BalanceRatio.get()}")
            command_output.send(writer)
            
            writer = echolib.MessageWriter()
            writer.writeString(f"ExposureTime {vimba_camera.ExposureTime.get()}")
            command_output.send(writer)

            ##############################
            # Get and set max frame rate #
            ##############################

            range = vimba_camera.get_feature_by_name("AcquisitionFrameRate").get_range()
            fr = floor(range[1])

            print(f"Frame rate range: [{range[0]}, {range[1]}]: setting {fr}")

            try:
                vimba_camera.AcquisitionFrameRate.set(fr)
            except Exception as e:
                    print(f"[ERROR] Setting camera frame rate failed: {e}")  
            
            ###########################
            # Send completed frames  #
            ###########################

            vimba_camera.start_streaming(handler.frame_handler, buffer_count = 100)

            while loop.wait(1):
                if handler.frame is not None:
                    output.send(Frame(image = handler.frame))

                    handler.frame = None
#                    print("got frame")
#                else:
#                    print("handler frame is none")
                if handler.message is not None:
                    writer = echolib.MessageWriter()
                    writer.writeString(handler.message)
                    command_output.send(writer)
                    handler.message = None


            ###########################
        
            vimba_camera.stop_streaming()
        
        print("Stopped streaming?")

if __name__=='__main__':
    main()
