import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import sys 
sys.path.append("..") 
from utils.geometries import get_K_crop_resize
import random

def hnormalized(vector):
    hnormalized_vector = (vector / vector[-1])[:-1]
    return hnormalized_vector

def crop_to_aspect_ratio(images, box, masks=None, K=None):
    assert images.dim() == 4
    bsz, _, h, w = images.shape
    assert box.dim() == 1
    assert box.shape[0] == 4
    w_output, h_output = box[[2, 3]] - box[[0, 1]]
    boxes = torch.cat(
        (torch.arange(bsz).unsqueeze(1).to(box.device).float(), box.unsqueeze(0).repeat(bsz, 1).float()),
        dim=1).to(images.device)
    images = torchvision.ops.roi_pool(images, boxes, output_size=(h_output, w_output))
    if masks is not None:
        assert masks.dim() == 4
        masks = torchvision.ops.roi_pool(masks, boxes, output_size=(h_output, w_output))
    if K is not None:
        assert K.dim() == 3
        assert K.shape[0] == bsz
        K = get_K_crop_resize(K, boxes[:, 1:], orig_size=(h, w), crop_resize=(h_output, w_output))
    return images, masks, K


def make_detections_from_segmentation(masks):
    detections = []
    if masks.dim() == 4:
        assert masks.shape[0] == 1
        masks = masks.squeeze(0)

    for mask_n in masks:
        dets_n = dict()
        for uniq in torch.unique(mask_n, sorted=True):
            ids = np.where((mask_n == uniq).cpu().numpy())
            x1, y1, x2, y2 = np.min(ids[1]), np.min(ids[0]), np.max(ids[1]), np.max(ids[0])
            dets_n[int(uniq.item())] = torch.tensor([x1, y1, x2, y2]).to(mask_n.device)
        detections.append(dets_n)
    return detections


def make_masks_from_det(detections, h, w):
    n_ids = len(detections)
    detections = torch.as_tensor(detections)
    masks = torch.zeros((n_ids, h, w)).byte()
    for mask_n, det_n in zip(masks, detections):
        x1, y1, x2, y2 = det_n.cpu().int().tolist()
        mask_n[y1:y2, x1:x2] = True
    return masks

def get_bbox(bbox,w,h, strict=True):
    assert len(bbox)==4
    wmin, hmin, wmax, hmax = bbox
    if wmax<0 or hmax <0 or wmin > w or hmin > h:
        print("wmax",wmax,"hmax",hmax,"wmin",wmin,"hmin",hmin)
    wmin,hmin,wmax,hmax=max(0,wmin),max(0,hmin),min(w,wmax),min(h,hmax)
    wnew=wmax-wmin
    hnew=hmax-hmin
    wmin=int(max(0,wmin-0.3*wnew))
    wmax=int(min(w,wmax+0.3*wnew))
    hmin=int(max(0,hmin-0.3*hnew))
    hmax=int(min(h,hmax+0.3*hnew))
    wnew=wmax-wmin
    hnew=hmax-hmin
    
    if not strict:
        randomw = (random.random()-0.2)/2
        randomh = (random.random()-0.2)/2
        
        dwnew=randomw*wnew
        wmax+=dwnew/2
        wmin-=dwnew/2

        dhnew=randomh*hnew
        hmax+=dhnew/2
        hmin-=dhnew/2
        
        wmin=int(max(0,wmin))
        wmax=int(min(w,wmax))
        hmin=int(max(0,hmin))
        hmax=int(min(h,hmax))
        wnew=wmax-wmin
        hnew=hmax-hmin
    
    if wnew < 150:
        wmax+=75
        wmin-=75
    if hnew < 120:
        hmax+=60
        hmin-=60
        
    wmin,hmin,wmax,hmax=max(0,wmin),max(0,hmin),min(w,wmax),min(h,hmax)
    wmin,hmin,wmax,hmax=min(w,wmin),min(h,hmin),max(0,wmax),max(0,hmax)
    new_bbox = np.array([wmin,hmin,wmax,hmax])
    return new_bbox

def get_bbox_raw(bbox):
    assert len(bbox)==4
    wmin, hmin, wmax, hmax = bbox
    wnew=wmax-wmin
    hnew=hmax-hmin
    wmin=int(wmin-0.3*wnew)
    wmax=int(wmax+0.3*wnew)
    hmin=int(hmin-0.3*hnew)
    hmax=int(hmax+0.3*hnew)
    wnew=wmax-wmin
    hnew=hmax-hmin
    
    if wnew < 150:
        wmax+=75
        wmin-=75
    if hnew < 120:
        hmax+=60
        hmin-=60

    new_bbox = np.array([wmin,hmin,wmax,hmax])
    return new_bbox

def resize_image(image, bbox, mask, state, bbox_strict_bounded=None):
    #image as np.array
    wmin, hmin, wmax, hmax = bbox
    square_size =int(max(wmax - wmin, hmax - hmin))
    square_image = np.zeros((square_size, square_size, 3), dtype=np.uint8)

    x_offset = int((square_size - (wmax-wmin)) // 2)
    y_offset = int((square_size- (hmax-hmin)) // 2)
    
    square_image[y_offset:y_offset+(hmax-hmin), x_offset:x_offset+(wmax-wmin)] = image[hmin:hmax, wmin:wmax]
    
    keypoints=state['objects'][0]['keypoints_2d']
    
    for k in keypoints:
        k[1]-=hmin
        k[1]+=y_offset
        k[0]+=x_offset
        k[0]-=wmin
    if bbox_strict_bounded is not None:
        bbox_strict_bounded_new = bbox_strict_bounded[0]-wmin+x_offset, bbox_strict_bounded[1]-hmin+y_offset, \
                                bbox_strict_bounded[2]-wmin+x_offset, bbox_strict_bounded[3]-hmin+y_offset
        
    K = state['camera']['K']
    K[0, 2] -= (wmin-x_offset)
    K[1, 2] -= (hmin-y_offset)
    if bbox_strict_bounded is None:
        return square_image, mask, state
    else:
        return square_image, mask, state, bbox_strict_bounded_new

def tensor_to_image(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = Image.fromarray(image)
    return image

def process_truncation(image, bbox, mask, state, max_pad=[120, 120, 120, 120]):
    #image as np.array
    wmin, hmin, wmax, hmax = bbox
    if wmin > 0 and hmin > 0 and hmax<480 and wmax <640:
        return image, bbox, mask, state
    d_wmin, d_hmin, d_wmax, d_hmax = int(-wmin), int(-hmin), int(wmax-640), int(hmax-480)
    d_wmin, d_hmin, d_wmax, d_hmax = int(max(0,d_wmin)), int(max(0,d_hmin)), int(max(0,d_wmax)), int(max(0,d_hmax))
    #print(d_wmin, d_hmin, d_wmax, d_hmax)
    d_wmin, d_hmin, d_wmax, d_hmax = min(max_pad[0],d_wmin), min(max_pad[1],d_hmin),min(max_pad[2],d_wmax),min(max_pad[3],d_hmax)
    wmax, hmax = 640 + d_wmax, 480+ d_hmax
    wnew, hnew = 640+d_wmax+d_wmin,480+d_hmax+d_hmin
    
    #print(wnew,hnew)
    new_image = np.zeros((hnew, wnew, 3), dtype=np.uint8)

    #print("d_hmin:",d_hmin,d_hmax, d_wmin, d_wmax,wnew, hnew,"hmax:",hmax)    
    new_image[d_hmin:d_hmin+480, d_wmin:d_wmin+640] = image[0:480, 0:640]
    
    
    keypoints=state['objects'][0]['keypoints_2d']
    
    for k in keypoints:
        k[1]+=d_hmin
        k[0]+=d_wmin
        
    K = state['camera']['K']
    K[0, 2] += (d_wmin)
    K[1, 2] += (d_hmin)
    
    # new_bbox = np.array([max(0,int(wmin + d_wmin)),max(0,int(hmin + d_hmin)),int(wmax + d_wmin),int(hmax + d_hmin)])
    bbox_raw = np.concatenate([np.min(keypoints, axis=0)[0:2], np.max(keypoints, axis=0)[0:2]])
    new_bbox = get_bbox(bbox_raw,wnew,hnew)
    return new_image, new_bbox, mask, state

def process_padding(image, bbox, mask, state, padding_pixel=25):
    #image as np.array
    keypoints=state['objects'][0]['keypoints_2d']
    # in_frame = 0
    # for k in keypoints:
    #     if k[0]>0 and k[0]<256 and k[1]>0 and k[1]<256:
    #         in_frame +=1
    # if in_frame ==7:
    #     return image, bbox, mask, state
    # d_pad = 30 - 3*in_frame
    d_pad = padding_pixel
    d_wmin, d_hmin, d_wmax, d_hmax = d_pad,d_pad,d_pad,d_pad
    
    wnew, hnew = 320+d_wmax+d_wmin,320+d_hmax+d_hmin
    
    #print(wnew,hnew)
    new_image = np.zeros((hnew, wnew, 3), dtype=np.uint8)

    #print("d_hmin:",d_hmin,d_hmax, d_wmin, d_wmax,wnew, hnew,"hmax:",hmax)    
    new_image[d_hmin:d_hmin+320, d_wmin:d_wmin+320] = image[0:320, 0:320]
    
    for k in keypoints:
        k[1]+=d_hmin
        k[0]+=d_wmin
        
    K = state['camera']['K']
    K[0, 2] += (d_wmin)
    K[1, 2] += (d_hmin)
    
    # new_bbox = np.array([max(0,int(wmin + d_wmin)),max(0,int(hmin + d_hmin)),int(wmax + d_wmin),int(hmax + d_hmin)])
    bbox_raw = np.concatenate([np.min(keypoints, axis=0)[0:2], np.max(keypoints, axis=0)[0:2]])
    new_bbox = get_bbox(bbox_raw,wnew,hnew)
    return new_image, new_bbox, mask, state

def bbox_transform(bbox, K_original_inv, K, resize_hw):
    wmin, hmin, wmax, hmax = bbox
    corners = np.array([[wmin, hmin, 1.0],
                        [wmax, hmin, 1.0],
                        [wmax, hmax, 1.0],
                        [wmin, hmax, 1.0]])
    corners3d_ill = np.matmul(K_original_inv, corners.T)
    new_corners = np.matmul(K, corners3d_ill).T
    assert all(new_corners[:,2] == 1.0), new_corners
    new_bbox = np.array([
                        np.clip(new_corners[0,0], 0, resize_hw[0]),
                        np.clip(new_corners[0,1], 0, resize_hw[1]),
                        np.clip(new_corners[1,0], 0, resize_hw[0]),
                        np.clip(new_corners[2,1], 0, resize_hw[1]),
                        ])
    return new_bbox

def get_extended_bbox(bbox, dwmin, dhmin, dwmax, dhmax, bounded=True, image_size=None):
    wmin, hmin, wmax, hmax = bbox
    extended_bbox = np.array([wmin-dwmin, hmin-dhmin, wmax+dwmax, hmax+dhmax])
    wmin, hmin, wmax, hmax = extended_bbox
    if bounded:
        assert image_size
        extended_bbox = np.array([max(0,wmin),max(0,hmin),min(image_size[0],wmax),min(image_size[1],hmax)])
    else:
        pass
    return extended_bbox
