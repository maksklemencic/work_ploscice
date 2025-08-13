#!/usr/bin/env python3

from math import floor
import numpy as np
from json import load

import echolib

from vimba import Vimba, PixelFormat, FrameStatus, Camera

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

    def __init__(self):

        self.frame = None
        self.n_frames = 0

        self.camera: Camera = None

        self.command_chain = []

        self.commands = \
        {
            "ExposureAuto":     lambda camera, v: camera.ExposureAuto.set(v),
            "BalanceWhiteAuto": lambda camera, v: camera.BalanceWhiteAuto.set(v),
            "BalanceRatio":     lambda camera, v: camera.BalanceRatio.set(v),
            "ExposureTime":     lambda camera, v: camera.ExposureTime.set(v)
        }

    def frame_handler(self, camera, frame):
        import time
#        print("frame handler start",time.time(), flush=True)
        if frame.get_status() == FrameStatus.Complete:
#            print("frame completed",time.time(), flush=True)
            frame.convert_pixel_format(PixelFormat.Rgb8)
            frame_copy = frame.as_numpy_ndarray()
            while self.frame != None:
                loop.wait(1)
            self.frame = np.array(frame_copy)

            self.n_frames += 1
        else:
            print("frame incomplete", flush=True)
#        print("queuing frame", flush=True)
        camera.queue_frame(frame)
#        print("handler done", flush=True)

    def callback_camera_input(self, message):

        command = echolib.MessageReader(message).readString().split(" ")

        value   = command[1]
        command = command[0]

        try:
            print(f"Got command: {command} -> {value}")

            if command == "ExposureAuto":
                self.commands[command](self.camera, value)
            elif command == "BalanceWhiteAuto":
                self.commands[command](self.camera, value)
            elif command == "BalanceRatio":
                value = float(value)

                self.commands[command](self.camera, value)
            elif command == "ExposureTime":
                value = float(value)

                self.commands[command](self.camera, value)

            self.camera.stop_streaming()

            self.camera.AcquisitionFrameRateEnable.set(True)
            range = self.camera.get_feature_by_name("AcquisitionFrameRate").get_range()
            fr = floor(range[1])

            print(f"Frame rate range: [{range[0]}, {range[1]}]: setting {fr}")

            try:
                self.camera.AcquisitionFrameRate.set(fr)
            except Exception as e:
                    print(f"[ERROR] Setting camera frame rate failed: {e}") 

            self.camera.start_streaming(self.frame_handler, buffer_count = 1) 

        except Exception as e:
            print(f"[ERROR] Error while setting {command} = {value} | error print -> {e}")
            

def setup_args():
    import argparse
    parser = argparse.ArgumentParser(prog = 'Vimba Camera Streamer')
    parser.add_argument('--config', default='/opt/config/camera0.json', help='Path to camera configuration json file')  
    
    args = parser.parse_args()
    
    return args

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
        "BinningHorizontal": lambda camera, v: camera.BinningHorizontal.set(int(v)),
        "BinningHorizontalMode": lambda camera, v: camera.BinningHorizontalMode.set(v),        
        "BinningVertical": lambda camera, v: camera.BinningVertical.set(int(v)),
        "BinningVerticalMode": lambda camera, v: camera.BinningVerticalMode.set(v)
    }

    ###########################

    handler = VimbaCameraHandler()

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

            output_ranges = None
            for f in ["BalanceRatio", "ExposureTime"]:
                range = vimba_camera.get_feature_by_name(f).get_range()

                output_ranges = f"{f} {range[0]} {range[1]}" if output_ranges is None else f"{output_ranges} {f} {range[0]} {range[1]}" 

            writer = echolib.MessageWriter()
            writer.writeString(output_ranges)

            command_output.send(writer)

            ###########################
            # Set default camera values
            ###########################

            for feature in config_set_functions:
                if feature in config:
                    try:
                        print(f"Setting default feature {feature} with {config[feature]}")

                        config_set_functions[feature](vimba_camera, config[feature])
                    except Exception as e:
                        print(f"[ERROR] Setting camera feature {feature} failer: {e}")

            ##############################
            # Get and set max frame rate #
            ##############################

            range = vimba_camera.get_feature_by_name("AcquisitionFrameRate").get_range()
            fr = floor(range[1]) if range[1] > 1 else range[1]

            print(f"Frame rate range: [{range[0]}, {range[1]}]: setting {fr}")

            try:
                vimba_camera.AcquisitionFrameRate.set(fr)
            except Exception as e:
                    print(f"[ERROR] Setting camera frame rate failed: {e}")  
            
            vimba_camera.BinningHorizontal.set(2)
            vimba_camera.BinningVertical.set(2)
            
            ###########################
            # Send compleated frames  #
            ###########################
            try:
                vimba_camera.start_streaming(handler.frame_handler, buffer_count = 1)

                while loop.wait(1):
                    if handler.frame is not None:
#                        from skimage.transform import resize
#                        print("sending frame", handler.frame.shape)
#                        handler.frame = np.array(resize(handler.frame, (1024, 1024)))
                        print("sending frame", handler.frame.shape)
                        output.send(Frame(image = handler.frame))
#                        ff = np.zeros((3672, 5496 ,3),dtype=np.uint8)
#                        ff = np.zeros((3036, 4024 ,3),dtype=np.uint8)
#                        output.send(Frame(image = ff))
                        
                        handler.frame = None
#                        print("got frame")
#                    else:
#                        print("handler frame is none")


            ###########################
            finally:
                vimba_camera.stop_streaming()
        
        print("Stoped streming?")

if __name__=='__main__':
    main()
