"""This backend now only support lidar. camera is no longer supported.
"""

import base64
import datetime
import io as sysio
import json
import pickle
import time
from pathlib import Path

import fire
import torch
import numpy as np
import skimage
from flask import Flask, jsonify, request
from flask_cors import CORS
from google.protobuf import text_format
from skimage import io

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])

# from second.data import kitti_common as kitti
# from second.data.all_dataset import get_dataset_class
# from second.protos import pipeline_pb2
# from second.pytorch.builder import (box_coder_builder, input_reader_builder,
#                                     lr_scheduler_builder, optimizer_builder,
#                                     second_builder)
# from second.pytorch.train import build_network, example_convert_to_torch

from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
import pcdet.datasets.kitti.kitti_object_eval_python.kitti_common as kitti
from pcdet.models import build_network, load_data_to_gpu


app = Flask("second")
CORS(app)

class SecondBackend:
    def __init__(self):
        self.root_path = None
        self.image_idxes = None
        self.dt_annos = None
        self.dataset = None
        self.net = None
        self.device = None
        self.logger = common_utils.create_logger()
        self.cfg = None


BACKEND = SecondBackend()

def error_response(msg):
    response = {}
    response["status"] = "error"
    response["message"] = "[ERROR]" + msg
    print("[ERROR]" + msg)
    return response


@app.route('/api/readinfo', methods=['POST'])
def readinfo():
    global BACKEND
    instance = request.json
    root_path = Path(instance["root_path"])
    response = {"status": "normal"}
    BACKEND.root_path = root_path
    info_path = Path(instance["info_path"])
    dataset_class_name = instance["dataset_class_name"]
    # ignore
    cfg_file = "cfgs/kitti_models/second.yaml"
    cfg_from_yaml_file(cfg_file, cfg)
    BACKEND.cfg = cfg

    root_path = "../data/kitti/"

    BACKEND.dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(root_path), logger=BACKEND.logger
    )

    # BACKEND.dataset = get_dataset_class(dataset_class_name)(root_path=root_path, info_path=info_path)
    BACKEND.image_idxes = list(range(len(BACKEND.dataset)))

    response["image_indexes"] = BACKEND.image_idxes
    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response

@app.route('/api/read_detection', methods=['POST'])
def read_detection():
    global BACKEND
    instance = request.json
    det_path = Path(instance["det_path"])
    response = {"status": "normal"}
    if BACKEND.root_path is None:
        return error_response("root path is not set")
    if Path(det_path).is_file():
        with open(det_path, "rb") as f:
            dt_annos = pickle.load(f)
    else:
        dt_annos = kitti.get_label_annos(det_path)
    BACKEND.dt_annos = dt_annos
    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response


@app.route('/api/get_pointcloud', methods=['POST'])
def get_pointcloud():
    global BACKEND
    instance = request.json
    response = {"status": "normal"}
    if BACKEND.root_path is None:
        return error_response("root path is not set")
    image_idx = instance["image_idx"]
    enable_int16 = instance["enable_int16"]

    idx = BACKEND.image_idxes.index(image_idx)

    # sensor_data = BACKEND.dataset.get_sensor_data(idx)
    data_dict = BACKEND.dataset[idx]

    # img_shape = image_info["image_shape"] # hw
    # if 'annotations' in sensor_data["lidar"]:
    if True:
        # annos = sensor_data["lidar"]['annotations']
        gt_boxes = data_dict["gt_boxes"].copy()
        # gt_boxes = annos["boxes"].copy()
        response["locs"] = gt_boxes[:, :3].tolist()
        response["dims"] = gt_boxes[:, 3:6].tolist()
        rots = np.concatenate([np.zeros([gt_boxes.shape[0], 2], dtype=np.float32), -gt_boxes[:, 6:7]], axis=1)
        response["rots"] = rots.tolist()
        # response["labels"] = annos["names"].tolist()
        response['labels'] = [BACKEND.cfg.CLASS_NAMES[int(i-1)] for i in gt_boxes[:,-1]]

    # response["num_features"] = sensor_data["lidar"]["points"].shape[1]
    response["num_features"] = 3
    points = data_dict["points"][:, :3]
    if enable_int16:
        int16_factor = instance["int16_factor"]
        points *= int16_factor
        points = points.astype(np.int16)
    pc_str = base64.b64encode(points.tobytes())
    response["pointcloud"] = pc_str.decode("utf-8")

    # if "score" in annos:
    #     response["score"] = score.tolist()
    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    print("send response with size {}!".format(len(pc_str)))
    return response

@app.route('/api/get_image', methods=['POST'])
def get_image():
    global BACKEND
    instance = request.json
    response = {"status": "normal"}
    # if BACKEND.root_path is None:
    #     return error_response("root path is not set")

    image_idx = instance["image_idx"]
    idx = BACKEND.image_idxes.index(image_idx)

    image = BACKEND.dataset.get_image_shape(BACKEND.dataset[idx]['frame_id'])

    if image is not None:
    # if "cam" in sensor_data and "data" in sensor_data["cam"] and sensor_data["cam"]["data"] is not None:
        # image_str = sensor_data["cam"]["data"]
        image_str = image
        response["image_b64"] = base64.b64encode(image_str).decode("utf-8")
        # response["image_b64"] = 'data:image/{};base64,'.format(sensor_data["cam"]["datatype"]) + response["image_b64"]
        response["image_b64"] = 'data:image/{};base64,'.format('png') + response["image_b64"]
        print("send an image with size {}!".format(len(response["image_b64"])))
    else:
        response["image_b64"] = ""
    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response

@app.route('/api/build_network', methods=['POST'])
def build_network_():
    global BACKEND
    instance = request.json
    cfg_path = Path(instance["config_path"])
    ckpt_path = Path(instance["checkpoint_path"])
    response = {"status": "normal"}
    if BACKEND.root_path is None:
        return error_response("root path is not set")
    if not cfg_path.exists():
        return error_response("config file not exist.")
    if not ckpt_path.exists():
        return error_response("ckpt file not exist.")
    # config = pipeline_pb2.TrainEvalPipelineConfig()

    model = build_network(model_cfg=BACKEND.cfg.MODEL, num_class=len(BACKEND.cfg.CLASS_NAMES),
        dataset= BACKEND.dataset)

    device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device).eval()

    model.load_params_from_file(filename=ckpt_path, logger=BACKEND.logger)
    # net.load_state_dict(torch.load(ckpt_path))

    # eval_input_cfg = config.eval_input_reader
    # BACKEND.dataset = input_reader_builder.build(
    #     eval_input_cfg,
    #     config.model.second,
    #     training=False,
    #     voxel_generator=net.voxel_generator,
    #     target_assigner=net.target_assigner).dataset

    BACKEND.model = model
    # BACKEND.config = config
    BACKEND.device = device
    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    print("build_network successful!")
    return response


@app.route('/api/inference_by_idx', methods=['POST'])
def inference_by_idx():
    global BACKEND
    cfg = BACKEND.cfg
    instance = request.json
    response = {"status": "normal"}
    if BACKEND.root_path is None:
        return error_response("root path is not set")

    metric = {
        'gt_num' : 0,
    }

    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    image_idx = instance["image_idx"]
    # remove_outside = instance["remove_outside"]
    idx = BACKEND.image_idxes.index(image_idx)

    data_dict = BACKEND.dataset[idx]
    # example = BACKEND.dataset[idx]
    data_dict = BACKEND.dataset.collate_batch([data_dict])
    load_data_to_gpu(data_dict)

    with torch.no_grad():
        pred_dicts, ret_dict = BACKEND.model(data_dict)

    disp_dict = {}
    statistics_info(cfg, ret_dict, metric, disp_dict)

    annos = BACKEND.dataset.generate_prediction_dicts(
        data_dict, pred_dicts, BACKEND.dataset.class_names, None
    )


    # don't forget to pad batch idx in coordinates
    # example["coordinates"] = np.pad(
    #     example["coordinates"], ((0, 0), (1, 0)),
    #     mode='constant',
    #     constant_values=0)


    # don't forget to add newaxis for anchors
    # example["anchors"] = example["anchors"][np.newaxis, ...]
    # example_torch = example_convert_to_torch(example, device=BACKEND.device)
    # pred = BACKEND.net(example_torch)[0]

    # box3d = pred["box3d_lidar"].detach().cpu().numpy()
    box3d = annos[0]['boxes_lidar']
    locs = box3d[:, :3]
    dims = box3d[:, 3:6]
    rots = np.concatenate([np.zeros([locs.shape[0], 2], dtype=np.float32), -box3d[:, 6:7]], axis=1)
    response["dt_locs"] = locs.tolist()
    response["dt_dims"] = dims.tolist()
    response["dt_rots"] = rots.tolist()
    response["dt_labels"] = annos[0]["name"].tolist()
    response["dt_scores"] = annos[0]["score"].tolist()

    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response


def main(port=16666):
    app.run(host='127.0.0.1', threaded=True, port=port)

if __name__ == '__main__':
    fire.Fire()
