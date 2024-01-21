from .augmentations import (
    PillowBlur, PillowSharpness, PillowContrast, PillowBrightness, PillowColor, to_torch_uint8,
    occlusion_aug, CropResizeToAspectAugmentation
)

rgb_augmentations=[
    PillowSharpness(p=0.3, factor_interval=(0., 50.)),
    PillowContrast(p=0.3, factor_interval=(0.7, 1.8)),
    PillowBrightness(p=0.3, factor_interval=(0.7, 1.8)),
    PillowColor(p=0.3, factor_interval=(0., 4.))
    ]

KEYPOINT_NAMES={ 
    'panda' : [
        'panda_link0', 'panda_link2', 'panda_link3',
        'panda_link4', 'panda_link6', 'panda_link7',
        'panda_hand'
    ],
    'baxter': [
        'torso_t0', 'right_s0','left_s0', 'right_s1', 'left_s1',
        'right_e0','left_e0', 'right_e1','left_e1','right_w0', 'left_w0',
        'right_w1','left_w1','right_w2', 'left_w2','right_hand','left_hand'
    ],
    'kuka' : [
        'iiwa7_link_0', 'iiwa7_link_1',
        'iiwa7_link_2', 'iiwa7_link_3',
        'iiwa7_link_4', 'iiwa7_link_5',
        'iiwa7_link_6', 'iiwa7_link_7'
    ],
    'owi535' :[
       'Rotation', 'Base', 'Elbow', 'Wrist'
    ]
}

KEYPOINT_NAMES_TO_LINK_NAMES = {
    "panda" : dict(zip(KEYPOINT_NAMES['panda'],KEYPOINT_NAMES['panda'])),
    "kuka" : {
        'iiwa7_link_0':'iiwa_link_0', 'iiwa7_link_1':'iiwa_link_1',
        'iiwa7_link_2':'iiwa_link_2', 'iiwa7_link_3':'iiwa_link_3',
        'iiwa7_link_4':'iiwa_link_4', 'iiwa7_link_5':'iiwa_link_5',
        'iiwa7_link_6':'iiwa_link_6', 'iiwa7_link_7':'iiwa_link_7'
    },
    "baxter" : {
        'torso_t0':'torso', 
        'right_s0':'right_upper_shoulder', 'left_s0':'left_upper_shoulder',
        'right_s1':'right_lower_shoulder', 'left_s1':'left_lower_shoulder',
        'right_e0':'right_upper_elbow','left_e0':'left_upper_elbow', 
        'right_e1':'right_lower_elbow','left_e1':'left_lower_elbow',
        'right_w0':'right_upper_forearm', 'left_w0':'left_upper_forearm',
        'right_w1':'right_lower_forearm', 'left_w1':'left_lower_forearm',
        'right_w2':'right_wrist', 'left_w2':'left_wrist',
        'right_hand':'right_hand','left_hand':'left_hand'
    },
    "owi535" : {
        'Rotation':'Rotation', 'Base':'Base', 'Elbow':'Elbow', 'Wrist':'Wrist'
    }
    }

LINK_NAMES = {
    'panda': ['panda_link0', 'panda_link2', 'panda_link3', 'panda_link4', 
              'panda_link6', 'panda_link7', 'panda_hand'],
    'kuka': ['iiwa_link_0', 'iiwa_link_1', 'iiwa_link_2', 'iiwa_link_3', 
             'iiwa_link_4', 'iiwa_link_5', 'iiwa_link_6', 'iiwa_link_7'],
    'baxter': ['torso', 'right_upper_shoulder', 'left_upper_shoulder', 'right_lower_shoulder', 
               'left_lower_shoulder', 'right_upper_elbow', 'left_upper_elbow', 'right_lower_elbow', 
               'left_lower_elbow', 'right_upper_forearm', 'left_upper_forearm', 'right_lower_forearm', 
               'left_lower_forearm', 'right_wrist', 'left_wrist', 'right_hand', 'left_hand'],
    #'owi535': ["Base","Elbow","Wrist","Model","Model","Model","Model","Base","Base","Base","Base","Elbow","Elbow","Elbow","Elbow","Wrist","Wrist"],
    'owi535' :[
        'Rotation', 'Base', 'Elbow', 'Wrist'
    ]
}

JOINT_NAMES={
    'panda': ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 
            'panda_joint5', 'panda_joint6', 'panda_joint7', 'panda_finger_joint1'],
    'kuka': ['iiwa_joint_1', 'iiwa_joint_2', 'iiwa_joint_3', 'iiwa_joint_4', 
                'iiwa_joint_5', 'iiwa_joint_6', 'iiwa_joint_7'],
    'baxter': ['head_pan', 'right_s0', 'left_s0', 'right_s1', 'left_s1', 
               'right_e0', 'left_e0', 'right_e1', 'left_e1', 'right_w0', 
               'left_w0', 'right_w1', 'left_w1', 'right_w2', 'left_w2'],
    'owi535' :[
        'Rotation', 'Base', 'Elbow', 'Wrist'
    ]
}

JOINT_TO_KP = {
    'panda': [1, 1, 2, 3, 4, 4, 5, 6],
    'kuka':[1,2,3,4,5,6,7],
    'baxter':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    'owi535':[0,1,2,3]  
}

# flip_pairs=[
#                 ["right_s0","left_s0"],["right_s1","left_s1"],["right_e0","left_e0"],
#                 ["right_e1","left_e1"],["right_w0","left_w0"],["right_w1","left_w1"],               
#                 ["right_w2","left_w2"],["right_hand","left_hand"]
#             ]
flip_pairs = [ [1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16] ]

PANDA_LIMB_LENGTH ={
    "link0-link2" : 0.3330,
    "link2-link3" : 0.3160,
    "link3-link4" : 0.0825,
    "link4-link6" : 0.39276,
    "link6-link7" : 0.0880,
    "link7-hand" : 0.1070
}
KUKA_LIMB_LENGTH ={
    "link0-link1" : 0.1500,
    "link1-link2" : 0.1900,
    "link2-link3" : 0.2100,
    "link3-link4" : 0.1900,
    "link4-link5" : 0.2100,
    "link5-link6" : 0.19946,
    "link6-link7" : 0.10122
}

LIMB_LENGTH = {
    "panda": list(PANDA_LIMB_LENGTH.values()),
    "kuka": list(KUKA_LIMB_LENGTH.values())
}

INITIAL_JOINT_ANGLE = {
    "zero": {
        "panda": {
            "panda_joint1": 0.0, 
            "panda_joint2": 0.0, 
            "panda_joint3": 0.0, 
            "panda_joint4": 0.0,
            "panda_joint5": 0.0, 
            "panda_joint6": 0.0, 
            "panda_joint7": 0.0, 
            "panda_finger_joint1": 0.0
        }, 
        "kuka": {
            "iiwa_joint_1": 0.0, 
            "iiwa_joint_2": 0.0, 
            "iiwa_joint_3": 0.0, 
            "iiwa_joint_4": 0.0, 
            "iiwa_joint_5": 0.0, 
            "iiwa_joint_6": 0.0, 
            "iiwa_joint_7": 0.0
        }, 
        "baxter": {
            "head_pan": 0.0, 
            "right_s0": 0.0, 
            "left_s0": 0.0, 
            "right_s1": 0.0, 
            "left_s1": 0.0, 
            "right_e0": 0.0,
            "left_e0": 0.0, 
            "right_e1": 0.0,
            "left_e1": 0.0, 
            "right_w0": 0.0,
            "left_w0": 0.0,
            "right_w1": 0.0, 
            "left_w1": 0.0, 
            "right_w2": 0.0,
            "left_w2": 0.0
        },
        "owi535":{
            "Rotation":0.0,
            "Base":0.0,
            "Elbow":0.0,
            "Wrist":0.0
        }
    }, 
    "mean": {
        "panda": {
            "panda_joint1": 0.0, 
            "panda_joint2": 0.0, 
            "panda_joint3": 0.0, 
            "panda_joint4": -1.52715, 
            "panda_joint5": 0.0, 
            "panda_joint6": 1.8675, 
            "panda_joint7": 0.0, 
            "panda_finger_joint1": 0.02
        }, 
        "kuka": {
            "iiwa_joint_1": 0.0, 
            "iiwa_joint_2": 0.0, 
            "iiwa_joint_3": 0.0, 
            "iiwa_joint_4": 0.0, 
            "iiwa_joint_5": 0.0, 
            "iiwa_joint_6": 0.0, 
            "iiwa_joint_7": 0.0
        }, 
        "baxter": {
            "head_pan": 0.0, 
            "right_s0": 0.0, 
            "left_s0": 0.0, 
            "right_s1": -0.5499999999999999, 
            "left_s1": -0.5499999999999999,
            "right_e0": 0.0,
            "left_e0": 0.0,
            "right_e1": 1.284,
            "left_e1": 1.284,
            "right_w0": 0.0, 
            "left_w0": 0.0,
            "right_w1": 0.2616018366049999,
            "left_w1": 0.2616018366049999,
            "right_w2": 0.0,
            "left_w2": 0.0      
        },
        "owi535":{
            "Rotation":0.0,
            "Base":-0.523598,
            "Elbow":0.523598,
            "Wrist":0.0
        }
    }
}

JOINT_BOUNDS = {
    "panda": [[-2.9671,  2.9671],
            [-1.8326,  1.8326],
            [-2.9671,  2.9671],
            [-3.1416,  0.0873],
            [-2.9671,  2.9671],
            [-0.0873,  3.8223],
            [-2.9671,  2.9671],
            [ 0.0000,  0.0400]],
    
    "kuka": [[-2.9671,  2.9671],
            [-2.0944,  2.0944],
            [-2.9671,  2.9671],
            [-2.0944,  2.0944],
            [-2.9671,  2.9671],
            [-2.0944,  2.0944],
            [-3.0543,  3.0543]],
    
    "baxter": [[-1.5708,  1.5708],
            [-1.7017,  1.7017],
            [-1.7017,  1.7017],
            [-2.1470,  1.0470],
            [-2.1470,  1.0470],
            [-3.0542,  3.0542],
            [-3.0542,  3.0542],
            [-0.0500,  2.6180],
            [-0.0500,  2.6180],
            [-3.0590,  3.0590],
            [-3.0590,  3.0590],
            [-1.5708,  2.0940],
            [-1.5708,  2.0940],
            [-3.0590,  3.0590],
            [-3.0590,  3.0590]],
    "owi535":[
        [-2.268928,2.268928],
        [-1.570796,1.047198],
        [-1.047198, 1.570796],
        [-0.785398,0.785398]
    ]
}


INTRINSICS_DICT = {
        "azure": (399.6578776041667, 399.4959309895833, 319.8955891927083, 244.0602823893229),
        "kinect": (525.0, 525.0, 319.5, 239.5),
        "realsense": (615.52392578125, 615.2191772460938, 328.2606506347656, 251.7917022705078),
        "orb": (615.52392578125, 615.2191772460938, 328.2606506347656, 251.7917022705078),
        
    }
