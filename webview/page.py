import os
import sys
import numpy as np
import pickle
import cv2
from flask import Flask, render_template, url_for, Blueprint, request, session
import config as cfg
from tools import gradcam
sys.path.append('..')
from utils.serialization import load_checkpoint
from models import define_net

cifar_page = Blueprint(
    'cifar_page', __name__,
    template_folder='templates',
    static_folder='static')

@cifar_page.route('/start')
def cifar_start():
    try:
        image_name = request.args.get("image_name", '')
    except:
        raise ValueError("To get 'mage_name' attribute encount error in Candidate")
    if image_name != '': 
        session['image_name'] = image_name
    images = np.loadtxt(cfg.test_image_list, dtype=np.str)
    image_info = []
    for image in images:
        image_info.append([os.path.splitext(image)[0].split('_')[-1], os.path.join(cfg.test_image_dir, image)])
    image_path = os.path.join(cfg.test_image_dir, image_name)
    return render_template("base.html", params = {'image_name': image_name, 
                                                'image_path': image_path, 
                                                'image_info': image_info[:cfg.candidate_num], 
                                                'images': images})

@cifar_page.route('/heatmap')
def cifar_heatmap():
    msg = {}
    msg['msg_type'] = 0
    msg['msg_info'] = ''
    if 'image_name' in session: 
        image_name = session['image_name']
    else:
        image_name = ''
        msg['msg_type'] = -1
        msg['msg_info'] = "The 'image_name' attribuare is empty! Please set in the page: "
    
    image_path = os.path.join(cfg.test_image_dir, image_name)
    img = gradcam.read_image(image_path)
    input_img = gradcam.preprocess_image(img)

    checkpoint = load_checkpoint(cfg.checkpoint_path)
    network = define_net(checkpoint['arch'], checkpoint['block_name'], 100).to(cfg.device)
    network.load_state_dict(checkpoint['state_dict'])

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    grad_cam = gradcam.GradCam(model=network, feature_module=network.layer3, target_layer_names=['11'], device=cfg.device)
    target_category = None
    grayscale_cam = grad_cam(input_img, target_category)

    grayscale_cam = cv2.resize(grayscale_cam, (img.shape[1], img.shape[0]))
    cv2.imwrite(os.path.join("static/heatmap", 'gradscale_' + image_name), grayscale_cam)
    cam = gradcam.post_cam_on_image(img / 255.0, grayscale_cam)
    cam_path = os.path.join("static/heatmap", 'cam_' + image_name)
    cv2.imwrite(cam_path, cam)

    return render_template("heatmap.html", params = {'msg': msg,
                                                    'image_name': image_name, 
                                                    'image_path': image_path,
                                                    'cam_name': 'cam_' + image_name,
                                                    'cam_path': cam_path})

@cifar_page.route('/nearest_neighbor_search')
def cifar_nearest_neighbor_search():
    msg = {}
    msg['msg_type'] = 0
    msg['msg_info'] = ''
    if 'image_name' in session: 
        image_name = session['image_name']
    else:
        image_name = ''
        msg['msg_type'] = -1
        msg['msg_info'] = "The 'image_name' attribuare is empty! Please set in the page: "

    try:
        topk = int(request.args.get("topk", 1))
    except:
        raise ValueError("To get 'TopK' attribute encount error!")

    images = np.loadtxt(cfg.test_image_list, dtype=np.str)
    tgt_idxs = np.array([])
    tgt_name = ''
    tgt_images_path = []
    if image_name != '':
        try:
            cand_idx = int(image_name.split('_')[0])
        except:
            raise ValueError("Parsing the 'image_name' attribute encount error, 'image_name': {}".format(image_name))
        test_embeddings = np.loadtxt(cfg.test_embeddings, dtype=np.float32)
        cand_embed = test_embeddings[cand_idx]

        dist = np.linalg.norm(cand_embed - test_embeddings, axis=1)
        tgt_idxs = np.argsort(dist)[:(topk+1)]
        tgt_name = ','.join(images[tgt_idxs[1:]].tolist())
        tgt_images_path = [os.path.join(cfg.test_image_dir, images[tgt_idx]) for tgt_idx in tgt_idxs[1:]]

    cand_image_path = os.path.join(cfg.test_image_dir, image_name)

    return render_template("nearest_neighbor_search.html", params = {'msg': msg,
                                                                    'image_name': image_name, 
                                                                    'cand_image_path': cand_image_path,
                                                                    'tgt_name': tgt_name,
                                                                    'tgt_images_path': tgt_images_path})
