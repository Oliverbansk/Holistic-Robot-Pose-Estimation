import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
os.environ["PYOPENGL_PLATFORM"] = "egl"
import cv2
import pyrender
import trimesh
from utils.transforms import point_projection_from_3d

IMAGENET_MEAN, IMAGENET_STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
CONNECTIVITY_DICT = {
    "panda": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
}
my_green = (135, 153, 124)
my_lightgreen = (187, 225, 163)
my_heavygreen = (150, 235, 120)
my_purple = (165, 175, 215)
my_orange = (230, 147, 115)
my_darkblue = (70, 80, 150)
my_lightblue = (140, 140, 255)
my_red = (255, 0, 0)
my_lightred = (255, 70, 70)

COLOR_DICT = {
    'panda': [
        my_lightblue, my_heavygreen, my_purple, my_green, my_lightblue, my_heavygreen,  my_purple 
    ],
    'gt':[
        my_lightred, my_lightred, my_lightred, my_lightred, my_lightred, my_lightred, my_lightred 
    ]
}


def denormalize_image(image):
    """Reverse to normalize_image() function"""
    return np.clip(255*(image * IMAGENET_STD + IMAGENET_MEAN), 0, 255)


def vis_joints_3d(batch_image, batch_joints, gt_joints, gt_2d, K_original, file_name="0.jpg", bbox=None,
                  errors=None, draw_skeleton=True, draw_original_image=False,
                  batch_image_path=None, batch_trans=None, nrow=4, ncol=8, size=10, 
                  padding=2, dataset_name='panda',vis_dir='vis/default_exp'):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    batch_size = batch_image.shape[0]
    batch_joints = batch_joints.reshape(batch_size, -1, 3)
    gt_2d_projection = point_projection_from_3d(K_original, gt_joints)
    pred_2d = point_projection_from_3d(K_original,batch_joints)
    plt.close('all')
    fig = plt.figure(figsize=(ncol*size, round(nrow*size*0.85)))
    plt.tight_layout()
    connectivity = CONNECTIVITY_DICT[dataset_name]

    for row in range(nrow):
        for col in range(ncol):
            
            # batch_idx = row*(ncol//4) + (col//4)
            batch_idx = row
            # print(batch_idx)
            if isinstance(batch_image, np.ndarray):
                img = batch_image[batch_idx]
                # img = denormalize_image(np.transpose(img.copy(), (1,2,0))).astype(np.uint8)   # C*H*W -> H*W*C
                img = (np.transpose(img.copy(), (1,2,0)) * 255).astype(np.uint8) # C*H*W -> H*W*C
            else:
                print("error")

            joints = batch_joints[batch_idx]
            gt = gt_joints[batch_idx]
            gt_2d_ = gt_2d_projection[batch_idx]
            pred_2d_ = pred_2d[batch_idx]
            if bbox is not None:
                assert bbox.shape == (batch_size, 4)
                bbox_ = bbox[batch_idx]

            if col%ncol == 0:  # draw image
                image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                x_, y_ = -1, -1
                for i in gt_2d_:
                    l=list(i)[0:2]
                    radius = 3
                    x=int(l[0])
                    y=int(l[1])
                    if x_ != -1:
                        cv2.line(image,(x,y),(x_,y_),(30,30,255),2)
                    cv2.circle(image,(x,y),4,(0,0,255),-2)
                    x_, y_ = x, y
                x_, y_ = -1, -1  
                for i in pred_2d_:
                    l=list(i)[0:2]
                    radius = 3
                    x=int(l[0])
                    y=int(l[1])
                    if x_ != -1:
                        cv2.line(image,(x,y),(x_,y_),(155,50,50),2)
                    cv2.circle(image,(x,y),3,(255,0,0),-2)
                    x_, y_ = x, y
                image_show = np.array(image)[:,:, : :-1]
                ax = fig.add_subplot(nrow, ncol, row * ncol + col + 1)
                ax.imshow(image_show)
                plt.title(f"error/ADD: {errors[batch_idx] :0.5f}m,  (prediction: blue, gt: red)")

            elif col%ncol == 1:   # draw prediction
                ax = fig.add_subplot(nrow, ncol, row * ncol + col + 1, projection='3d')
                ax.view_init(elev=12, azim=-70)
                ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1], s=25, c=np.array([my_darkblue])/255, edgecolors=np.array([my_darkblue])/255)
                if draw_skeleton:
                    for i, jt in enumerate(connectivity):
                        xs, ys, zs = [np.array([joints[jt[0], j], joints[jt[1], j]]) for j in range(3)]
                        color = COLOR_DICT[dataset_name][i]
                        if color == (0,139,139): # green
                            set_zorder = 2
                        elif color == (255,215,0): # yellow
                            set_zorder = 2
                        else:
                            set_zorder = 1
                        color = np.array(color) / 255
                        

                        ax.plot(xs, zs, ys, lw=4, ls='-', c=color, zorder=set_zorder, solid_capstyle='round')
                plt.xlim(-0.5,0.5)
                plt.ylim(0.5,2.0)
                ax.set_zlim(0.4,-0.5)
                if row == 0:
                    plt.title("prediction")

                if draw_original_image:
                    # extent = (0.0, 640.0, 0.0, 480.0)  
                    # ax.imshow(img, cmap='gray', extent=extent, origin='lower')
                    pass

            elif col%ncol == 2: # draw gt
                ax = fig.add_subplot(nrow, ncol, row * ncol + col + 1, projection='3d')
                
                ax.view_init(elev=12, azim=-70)
                ax.scatter(gt[:, 0], gt[:, 2], gt[:, 1], s=25, c=np.array([my_darkblue])/255, edgecolors=np.array([my_darkblue])/255)
                if draw_skeleton:
                    for i, jt in enumerate(connectivity):
                        xs, ys, zs = [np.array([gt[jt[0], j], gt[jt[1], j]]) for j in range(3)]
                        color = COLOR_DICT['gt'][i]
                        if color == (0,139,139): # green
                            set_zorder = 2
                        elif color == (255,215,0): # yellow
                            set_zorder = 2
                        else:
                            set_zorder = 1
                        color = np.array(color) / 255

                        ax.plot(xs, zs, ys, lw=4, ls='-', c=color, zorder=set_zorder, solid_capstyle='round')
                plt.xlim(-0.5,0.5)
                plt.ylim(0.5,2.0)
                ax.set_zlim(0.4,-0.5)
                if row == 0:
                    plt.title("gt")

                if draw_original_image:
                    # extent = (0.0, 640.0, 0.0, 480.0)  
                    # ax.imshow(img, cmap='gray', extent=extent, origin='lower')
                    pass

            elif col%ncol == 3: # draw prediction + gt
                ax = fig.add_subplot(nrow, ncol, row * ncol + col + 1, projection='3d')
                ax.view_init(elev=12, azim=-70)
                ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1], s=25, c=np.array([my_lightblue])/255, edgecolors=np.array([my_lightblue])/255)
                if draw_skeleton:
                    for i, jt in enumerate(connectivity):
                        xs, ys, zs = [np.array([joints[jt[0], j], joints[jt[1], j]]) for j in range(3)]
                        color = COLOR_DICT[dataset_name][i]
                        if color == (0,139,139): # green
                            set_zorder = 2
                        elif color == (255,215,0): # yellow
                            set_zorder = 2
                        else:
                            set_zorder = 1
                        color = np.array(color) / 255
                        ax.plot(xs, zs, ys, lw=3.5, ls='-', c=color, zorder=set_zorder, solid_capstyle='round')
                plt.xlim(-0.5,0.5)
                plt.ylim(0.5,2.0)
                ax.set_zlim(0.4,-0.5)
                ax.scatter(gt[:, 0], gt[:, 2], gt[:, 1], s=10, c=np.array([my_darkblue])/255, edgecolors=np.array([my_darkblue])/255)
                if draw_skeleton:
                    for i, jt in enumerate(connectivity):
                        xs, ys, zs = [np.array([gt[jt[0], j], gt[jt[1], j]]) for j in range(3)]
                        color = COLOR_DICT['gt'][i]
                        if color == (0,139,139): # green
                            set_zorder = 2
                        elif color == (255,215,0): # yellow
                            set_zorder = 2
                        else:
                            set_zorder = 1
                        color = np.array(color) / 255
                        ax.plot(xs, zs, ys, lw=2, ls='-', c=color, zorder=set_zorder, solid_capstyle='round')
                plt.xlim(-0.5,0.5)
                plt.ylim(0.5,2.0)
                ax.set_zlim(0.4,-0.5)
                if row == 0:
                    plt.title("prediction + gt")

            elif col%ncol == 4: # draw prediction + gt
                ax = fig.add_subplot(nrow, ncol, row * ncol + col + 1, projection='3d')
                ax.view_init(elev=12, azim=-40)
                ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1], s=25, c=np.array([my_lightblue])/255, edgecolors=np.array([my_lightblue])/255)
                if draw_skeleton:
                    for i, jt in enumerate(connectivity):
                        xs, ys, zs = [np.array([joints[jt[0], j], joints[jt[1], j]]) for j in range(3)]
                        color = COLOR_DICT[dataset_name][i]
                        if color == (0,139,139): # green
                            set_zorder = 2
                        elif color == (255,215,0): # yellow
                            set_zorder = 2
                        else:
                            set_zorder = 1
                        color = np.array(color) / 255
                        ax.plot(xs, zs, ys, lw=3.5, ls='-', c=color, zorder=set_zorder, solid_capstyle='round')
                plt.xlim(-0.5,0.5)
                plt.ylim(0.5,2.0)
                ax.set_zlim(0.4,-0.5)
                ax.scatter(gt[:, 0], gt[:, 2], gt[:, 1], s=10, c=np.array([my_darkblue])/255, edgecolors=np.array([my_darkblue])/255)
                if draw_skeleton:
                    for i, jt in enumerate(connectivity):
                        xs, ys, zs = [np.array([gt[jt[0], j], gt[jt[1], j]]) for j in range(3)]
                        color = COLOR_DICT['gt'][i]
                        if color == (0,139,139): # green
                            set_zorder = 2
                        elif color == (255,215,0): # yellow
                            set_zorder = 2
                        else:
                            set_zorder = 1
                        color = np.array(color) / 255
                        ax.plot(xs, zs, ys, lw=2, ls='-', c=color, zorder=set_zorder, solid_capstyle='round')
                plt.xlim(-0.5,0.5)
                plt.ylim(0.5,2.0)
                ax.set_zlim(0.4,-0.5)
                if row == 0:
                    plt.title("prediction + gt")

            elif col%ncol == 5: # draw prediction + gt
                ax = fig.add_subplot(nrow, ncol, row * ncol + col + 1, projection='3d')
                ax.view_init(elev=12, azim=0)
                ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1], s=25, c=np.array([my_lightblue])/255, edgecolors=np.array([my_lightblue])/255)
                if draw_skeleton:
                    for i, jt in enumerate(connectivity):
                        xs, ys, zs = [np.array([joints[jt[0], j], joints[jt[1], j]]) for j in range(3)]
                        color = COLOR_DICT[dataset_name][i]
                        if color == (0,139,139): # green
                            set_zorder = 2
                        elif color == (255,215,0): # yellow
                            set_zorder = 2
                        else:
                            set_zorder = 1
                        color = np.array(color) / 255
                        ax.plot(xs, zs, ys, lw=3.5, ls='-', c=color, zorder=set_zorder, solid_capstyle='round')
                plt.xlim(-0.5,0.5)
                plt.ylim(0.5,2.0)
                ax.set_zlim(0.4,-0.5)
                ax.scatter(gt[:, 0], gt[:, 2], gt[:, 1], s=10, c=np.array([my_darkblue])/255, edgecolors=np.array([my_darkblue])/255)
                if draw_skeleton:
                    for i, jt in enumerate(connectivity):
                        xs, ys, zs = [np.array([gt[jt[0], j], gt[jt[1], j]]) for j in range(3)]
                        color = COLOR_DICT['gt'][i]
                        if color == (0,139,139): # green
                            set_zorder = 2
                        elif color == (255,215,0): # yellow
                            set_zorder = 2
                        else:
                            set_zorder = 1
                        color = np.array(color) / 255
                        ax.plot(xs, zs, ys, lw=2, ls='-', c=color, zorder=set_zorder, solid_capstyle='round')
                plt.xlim(-0.5,0.5)
                plt.ylim(0.5,2.0)
                ax.set_zlim(0.4,-0.5)
                if row == 0:
                    plt.title("prediction + gt")

            elif col%ncol == 6: # draw prediction + gt
                ax = fig.add_subplot(nrow, ncol, row * ncol + col + 1, projection='3d')
                ax.view_init(elev=12, azim=20)
                ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1], s=25, c=np.array([my_lightblue])/255, edgecolors=np.array([my_lightblue])/255)
                if draw_skeleton:
                    for i, jt in enumerate(connectivity):
                        xs, ys, zs = [np.array([joints[jt[0], j], joints[jt[1], j]]) for j in range(3)]
                        color = COLOR_DICT[dataset_name][i]
                        if color == (0,139,139): # green
                            set_zorder = 2
                        elif color == (255,215,0): # yellow
                            set_zorder = 2
                        else:
                            set_zorder = 1
                        color = np.array(color) / 255
                        ax.plot(xs, zs, ys, lw=3.5, ls='-', c=color, zorder=set_zorder, solid_capstyle='round')
                plt.xlim(-0.5,0.5)
                plt.ylim(0.5,2.0)
                ax.set_zlim(0.4,-0.5)
                ax.scatter(gt[:, 0], gt[:, 2], gt[:, 1], s=10, c=np.array([my_darkblue])/255, edgecolors=np.array([my_darkblue])/255)
                if draw_skeleton:
                    for i, jt in enumerate(connectivity):
                        xs, ys, zs = [np.array([gt[jt[0], j], gt[jt[1], j]]) for j in range(3)]
                        color = COLOR_DICT['gt'][i]
                        if color == (0,139,139): # green
                            set_zorder = 2
                        elif color == (255,215,0): # yellow
                            set_zorder = 2
                        else:
                            set_zorder = 1
                        color = np.array(color) / 255
                        ax.plot(xs, zs, ys, lw=2, ls='-', c=color, zorder=set_zorder, solid_capstyle='round')
                plt.xlim(-0.5,0.5)
                plt.ylim(0.5,2.0)
                ax.set_zlim(0.4,-0.5)
                if row == 0:
                    plt.title("prediction + gt")

            elif col%ncol == 7: # draw prediction + gt
                ax = fig.add_subplot(nrow, ncol, row * ncol + col + 1, projection='3d')
                ax.view_init(elev=12, azim=50)
                ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1], s=25, c=np.array([my_lightblue])/255, edgecolors=np.array([my_lightblue])/255)
                if draw_skeleton:
                    for i, jt in enumerate(connectivity):
                        xs, ys, zs = [np.array([joints[jt[0], j], joints[jt[1], j]]) for j in range(3)]
                        color = COLOR_DICT[dataset_name][i]
                        if color == (0,139,139): # green
                            set_zorder = 2
                        elif color == (255,215,0): # yellow
                            set_zorder = 2
                        else:
                            set_zorder = 1
                        color = np.array(color) / 255
                        ax.plot(xs, zs, ys, lw=3.5, ls='-', c=color, zorder=set_zorder, solid_capstyle='round')
                plt.xlim(-0.5,0.5)
                plt.ylim(0.5,2.0)
                ax.set_zlim(0.4,-0.5)
                ax.scatter(gt[:, 0], gt[:, 2], gt[:, 1], s=10, c=np.array([my_darkblue])/255, edgecolors=np.array([my_darkblue])/255)
                if draw_skeleton:
                    for i, jt in enumerate(connectivity):
                        xs, ys, zs = [np.array([gt[jt[0], j], gt[jt[1], j]]) for j in range(3)]
                        color = COLOR_DICT['gt'][i]
                        if color == (0,139,139): # green
                            set_zorder = 2
                        elif color == (255,215,0): # yellow
                            set_zorder = 2
                        else:
                            set_zorder = 1
                        color = np.array(color) / 255
                        ax.plot(xs, zs, ys, lw=2, ls='-', c=color, zorder=set_zorder, solid_capstyle='round')
                plt.xlim(-0.5,0.5)
                plt.ylim(0.5,2.0)
                ax.set_zlim(0.4,-0.5)
                if row == 0:
                    plt.title("prediction + gt")

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')

    save_path = os.path.join(vis_dir, file_name)
    print(save_path)
    plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0.5)
    # plt.show()

    plt.close('all')
    return



def render_mesh(height, width, mesh, face, cam_param):
    # mesh
    mesh = trimesh.Trimesh(mesh, face)
    rot = trimesh.transformations.rotation_matrix(
	np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')
    
    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)
 
    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=1.0)
   
    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:,:,:3].astype(np.float32)
    # valid_mask = (depth > 0)[:,:,None]

    return rgb, depth


def vis_3dkp_single_view(preds, gt, save_path, elev=12, azim=0, error_val=None):
    assert len(preds.shape) == 2 and len(gt.shape) == 2, (preds.shape, gt.shape)
    preds = preds.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.view_init(elev=elev, azim=azim)
    connectivity = CONNECTIVITY_DICT["panda"]
    
    ax.scatter(preds[:, 0], preds[:, 2], preds[:, 1], s=25, c=np.array([my_lightblue])/255, edgecolors=np.array([my_lightblue])/255)
    for i, jt in enumerate(connectivity):
        xs, ys, zs = [np.array([preds[jt[0], j], preds[jt[1], j]]) for j in range(3)]
        color = COLOR_DICT["panda"][i]
        if color == (0,139,139): # green
            set_zorder = 2
        elif color == (255,215,0): # yellow
            set_zorder = 2
        else:
            set_zorder = 1
        color = np.array(color) / 255
        ax.plot(xs, zs, ys, lw=3.5, ls='-', c=color, zorder=set_zorder, solid_capstyle='round')
    plt.xlim(-0.5,0.5)
    plt.ylim(0.5,2.0)
    ax.set_zlim(0.4,-0.5)
    ax.scatter(gt[:, 0], gt[:, 2], gt[:, 1], s=10, c=np.array([my_darkblue])/255, edgecolors=np.array([my_darkblue])/255)
    for i, jt in enumerate(connectivity):
        xs, ys, zs = [np.array([gt[jt[0], j], gt[jt[1], j]]) for j in range(3)]
        color = COLOR_DICT['gt'][i]
        if color == (0,139,139): # green
            set_zorder = 2
        elif color == (255,215,0): # yellow
            set_zorder = 2
        else:
            set_zorder = 1
        color = np.array(color) / 255
        ax.plot(xs, zs, ys, lw=2, ls='-', c=color, zorder=set_zorder, solid_capstyle='round')
    plt.xlim(-0.5,0.5)
    plt.ylim(0.5,2.0)
    ax.set_zlim(0.4,-0.5)
    
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    
    if error_val is not None:
        plt.title(f"errors: {error_val :0.5f}m")
                
    plt.savefig(save_path, dpi=80, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close('all')