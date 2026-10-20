# Everything the detector found in ONE camera image.
# header: copied from the analysed image (same stamp and frame_id), so that TF lookups use the moment
#         the picture was taken, not the moment the detector finished.
std_msgs/Header header
uint32 image_width
uint32 image_height
ObjectDetection[] detections    # may be empty: "looked, saw nothing"
