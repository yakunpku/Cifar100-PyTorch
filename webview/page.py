import os
import numpy as np
import pickle
import cv2
from flask import Flask, render_template, url_for, Blueprint, request, session
import config as cfg

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
    try:
        image_name = request.args.get("image_name", '')
    except:
        raise ValueError("Get image name has error in heatmap")

    images = np.loadtxt(cfg.test_image_list, dtype=np.str)
    return render_template("heatmap.html", params = {'image_name': image_name, 'images': images[:cfg.candidate_num]})

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
            raise ValueError("Parsing the 'image_name' attribuate encount error, 'image_name': {}".format(image_name))
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