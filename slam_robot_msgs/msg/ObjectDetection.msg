# One object found in one camera image. Pixel coordinates, origin = top-left corner of the image.
string label          # class name, e.g. "red_box"
float32 confidence    # 0.0 .. 1.0
int32 x_min
int32 y_min
int32 x_max
int32 y_max
# True when the box touches the image border. The object is then partly outside the picture, so the
# centre of the box is NOT the centre of the object: do not use it for a 3D position.
bool truncated
