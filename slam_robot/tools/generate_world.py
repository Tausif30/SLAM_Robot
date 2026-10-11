import os
# Rooms: west room x in [-4, 0.4], east room x in [0.6, 4]; y in [-3, 3].
# The doorway in the partition is at y = 0.8 .. 2.0 (1.2 m wide).
# Robot start (set in simulation.launch.py): x=-3.0, y=-2.0.
#the red target box is in the east room, hidden behind the partition.

WALLS = [  # name, x, y, size_x, size_y
    ('wall_north', 0.0, 3.1, 8.4, 0.2),
    ('wall_south', 0.0, -3.1, 8.4, 0.2),
    ('wall_east', 4.1, 0.0, 0.2, 6.0),
    ('wall_west', -4.1, 0.0, 0.2, 6.0),
    ('partition_low', 0.5, -1.1, 0.2, 3.8),
    ('partition_high', 0.5, 2.5, 0.2, 1.0),
]
CRATES = [  # name, x, y, size_x, size_y, size_z, yaw
    ('crate_west', -3.0, 1.2, 0.6, 0.6, 0.8, 0.0),
    ('crate_east', 2.0, 0.0, 0.8, 0.5, 0.6, 0.4),
]
PILLARS = [  # name, x, y, radius, height
    ('pillar_west', -1.5, -0.5, 0.25, 1.0),
]
RED_BOX_POSE = (3.0, -2.0)    # target, hidden behind the partition
BLUE_BOX_POSE = (-1.0, 1.8)   # distractor, visible from the start

GREY, BROWN, DARK = '0.75 0.75 0.75 1', '0.55 0.40 0.25 1', '0.35 0.35 0.35 1'
RED, BLUE = '0.9 0.05 0.05 1', '0.05 0.1 0.9 1'


def material(c):
    return f'<material><ambient>{c}</ambient><diffuse>{c}</diffuse></material>'


def box_model(name, x, y, sx, sy, sz, yaw, color):
    g = f'<geometry><box><size>{sx} {sy} {sz}</size></box></geometry>'
    return f'''
    <model name="{name}">
      <static>true</static>
      <pose>{x} {y} {sz / 2:g} 0 0 {yaw}</pose>
      <link name="link">
        <collision name="collision">{g}</collision>
        <visual name="visual">{g}{material(color)}</visual>
      </link>
    </model>'''


def cylinder_model(name, x, y, r, h, color):
    g = f'<geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry>'
    return f'''
    <model name="{name}">
      <static>true</static>
      <pose>{x} {y} {h / 2:g} 0 0 0</pose>
      <link name="link">
        <collision name="collision">{g}</collision>
        <visual name="visual">{g}{material(color)}</visual>
      </link>
    </model>'''


def cube_include(name, pose):
    return f'''
    <include>
      <name>{name}</name>
      <uri>model://{name}</uri>
      <pose>{pose[0]} {pose[1]} 0 0 0 0</pose>
    </include>'''


def write_cube_model(root, name, color):
    d = os.path.join(root, 'models', name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'model.config'), 'w') as f:
        f.write(f'''<?xml version="1.0"?>
<model>
  <name>{name}</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
  <description>A 0.3 m cube used as a search target or distractor.</description>
</model>
''')
    g = '<geometry><box><size>0.3 0.3 0.3</size></box></geometry>'
    with open(os.path.join(d, 'model.sdf'), 'w') as f:
        f.write(f'''<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <pose>0 0 0.15 0 0 0</pose>
      <collision name="collision">{g}</collision>
      <visual name="visual">{g}{material(color)}</visual>
    </link>
  </model>
</sdf>
''')

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parts = ['''<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="object_search_world">

    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <scene>
      <ambient>0.6 0.6 0.6 1</ambient>
      <shadows>false</shadows>
    </scene>

    <gui>
      <camera name="user_camera">
        <pose>0 -0.8 12 0 1.5 1.5708</pose>
      </camera>
    </gui>

    <include><uri>model://sun</uri></include>
    <include><uri>model://ground_plane</uri></include>

    <!-- Walls and partition -->''']
    parts += [box_model(n, x, y, sx, sy, 1.0, 0, GREY) for n, x, y, sx, sy in WALLS]
    parts.append('\n    <!-- Obstacles -->')
    parts += [box_model(n, x, y, sx, sy, sz, yaw, BROWN)
              for n, x, y, sx, sy, sz, yaw in CRATES]
    parts += [cylinder_model(n, x, y, r, h, DARK) for n, x, y, r, h in PILLARS]
    parts.append('\n    <!-- Target (red) and distractor (blue), from models/ -->')
    parts += [cube_include('red_box', RED_BOX_POSE), cube_include('blue_box', BLUE_BOX_POSE)]
    parts.append('\n  </world>\n</sdf>\n')

    os.makedirs(os.path.join(root, 'worlds'), exist_ok=True)
    with open(os.path.join(root, 'worlds', 'object_search.world'), 'w') as f:
        f.write('\n'.join(parts))
    write_cube_model(root, 'red_box', RED)
    write_cube_model(root, 'blue_box', BLUE)
    print('Wrote worlds/object_search.world, models/red_box, models/blue_box')


if __name__ == '__main__':
    main()
