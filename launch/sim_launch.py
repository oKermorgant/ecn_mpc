from simple_launch import SimpleLauncher
from nav2_common.launch import RewrittenYaml


def generate_launch_description():

    sl = SimpleLauncher()
    manual = sl.declare_arg('manual', False)

    sl.include('zoe', 'sim_ecn_launch.py',
               launch_arguments={'display': False, 'rviz': False, 'manual': manual})

    sl.rviz(sl.find('ecn_mpc', 'zoe.rviz'))

    nav2_params = sl.find('ecn_mpc', 'nav2.yaml')
    configured_params = RewrittenYaml(
                            source_file=nav2_params,
                            root_key='zoe',
                            param_rewrites = {},
                            convert_types=True)
    remappings = {'/zoe/map': '/map', 'map': '/map', '/scan': 'scan'}

    with sl.group(ns = 'zoe'):
        sl.node('nav2_planner', 'planner_server', parameters = [configured_params],
                remappings = remappings)
        sl.node('nav2_lifecycle_manager','lifecycle_manager',name='lifecycle_manager',
                output='screen',
                parameters=[{'autostart': True,
                            'node_names': ['planner_server']}])

        sl.node('ecn_mpc', 'navigator')

    return sl.launch_description()
