docker run -it --rm \
    --device /dev/bus/usb/002 \
    -v ~/vicos-cube/cube-main/config/camera_allied.json:/opt/config/camera0.json \
    -v ~/vicos-cube/cube-cameras/allied_vision:/data/reference_condition \
    -v /tmp/echo.sock:/tmp/echo.sock \
    camerafeed_allied:v1