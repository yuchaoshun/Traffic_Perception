import numpy as np
import torchvision
import time
import math
import os
import copy
import pdb
import argparse
import sys
import cv2
import skimage.io
import skimage.transform
import skimage.color
import skimage
import torch
from models.networks.pose_dla_CoordAtt_dcn_a3dv6 import *
from torchvision.transforms import transforms as T
from torchvision.transforms import functional as F
from opts import opts
from models.decode import mot_decode
from utils.post_process import ctdet_post_process
from model_utils import *
from models.util import _tranpose_and_gather_feat
from utils.image import transform_preds

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer, \
    RGB_MEAN, RGB_STD
from scipy.optimize import linear_sum_assignment

# assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))

color_list = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (128, 0, 255),
              (0, 128, 255), (128, 255, 0), (0, 255, 128), (255, 128, 0), (255, 0, 128), (128, 128, 255),
              (128, 255, 128), (255, 128, 128), (128, 128, 0), (128, 0, 128)]
global hm
global time_sum


class detect_rect:
    def __init__(self):
        self.curr_frame = 0
        self.curr_rect = np.array([0, 0, 1, 1])
        self.next_rect = np.array([0, 0, 1, 1])
        self.conf = 0
        self.id = 0
        self.type = 0
        self.speed = 0.0

    @property
    def position(self):
        x = (self.curr_rect[0] + self.curr_rect[2]) / 2
        y = (self.curr_rect[1] + self.curr_rect[3]) / 2
        return np.array([x, y])

    @property
    def size(self):
        w = self.curr_rect[2] - self.curr_rect[0]
        h = self.curr_rect[3] - self.curr_rect[1]
        return np.array([w, h])


class tracklet:
    def __init__(self, det_rect):
        self.id = det_rect.id
        self.rect_list = [det_rect]
        self.rect_num = 1
        self.last_rect = det_rect
        self.last_frame = det_rect.curr_frame
        self.no_match_frame = 0

    def add_rect(self, det_rect):
        self.rect_list.append(det_rect)
        self.rect_num = self.rect_num + 1
        self.last_rect = det_rect
        self.last_frame = det_rect.curr_frame

    @property
    def velocity(self):
        if (self.rect_num < 2):
            return (0, 0)
        elif (self.rect_num < 6):
            return (self.rect_list[self.rect_num - 1].position - self.rect_list[self.rect_num - 2].position) / (
                        self.rect_list[self.rect_num - 1].curr_frame - self.rect_list[self.rect_num - 2].curr_frame)
        else:
            v1 = (self.rect_list[self.rect_num - 1].position - self.rect_list[self.rect_num - 4].position) / (
                        self.rect_list[self.rect_num - 1].curr_frame - self.rect_list[self.rect_num - 4].curr_frame)
            v2 = (self.rect_list[self.rect_num - 2].position - self.rect_list[self.rect_num - 5].position) / (
                        self.rect_list[self.rect_num - 2].curr_frame - self.rect_list[self.rect_num - 5].curr_frame)
            v3 = (self.rect_list[self.rect_num - 3].position - self.rect_list[self.rect_num - 6].position) / (
                        self.rect_list[self.rect_num - 3].curr_frame - self.rect_list[self.rect_num - 6].curr_frame)
            return (v1 + v2 + v3) / 3


def cal_iou(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    i_w = min(x2, x4) - max(x1, x3)
    i_h = min(y2, y4) - max(y1, y3)
    if (i_w <= 0 or i_h <= 0):
        return 0
    i_s = i_w * i_h
    s_1 = (x2 - x1) * (y2 - y1)
    s_2 = (x4 - x3) * (y4 - y3)
    return float(i_s) / (s_1 + s_2 - i_s)


def cal_simi(det_rect1, det_rect2):
    return cal_iou(det_rect1.next_rect, det_rect2.curr_rect)


def cal_simi_track_det(track, det_rect):
    if (det_rect.curr_frame <= track.last_frame):
        print("cal_simi_track_det error")
        return 0
    elif (det_rect.curr_frame - track.last_frame == 1):
        return cal_iou(track.last_rect.next_rect, det_rect.curr_rect)
    else:
        pred_rect = track.last_rect.curr_rect + np.append(track.velocity, track.velocity) * (
                    det_rect.curr_frame - track.last_frame)
        return cal_iou(pred_rect, det_rect.curr_rect)


def track_det_match(tracklet_list, det_rect_list, min_iou=0.5):
    num1 = len(tracklet_list)
    num2 = len(det_rect_list)
    cost_mat = np.zeros((num1, num2))
    for i in range(num1):
        for j in range(num2):
            cost_mat[i, j] = -cal_simi_track_det(tracklet_list[i], det_rect_list[j])

    match_result = linear_sum_assignment(cost_mat)
    match_result = np.asarray(match_result)
    match_result = np.transpose(match_result)

    matches, unmatched1, unmatched2 = [], [], []
    for i in range(num1):
        if i not in match_result[:, 0]:
            unmatched1.append(i)
    for j in range(num2):
        if j not in match_result[:, 1]:
            unmatched2.append(j)
    for i, j in match_result:
        if cost_mat[i, j] > -min_iou:
            unmatched1.append(i)
            unmatched2.append(j)
        else:
            matches.append((i, j))
    return matches, unmatched1, unmatched2


def draw_caption(image, box, caption, color):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 8), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)


def post_process(dets, meta, opt, dis, speed):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets_next = []
    for i in range(dets.shape[0]):
        dets_next.append(np.array([(dets[i, :, 0] + dis[:, 0]), (dets[i, :, 1] + dis[:, 1]), (dets[i, :, 2] + dis[:, 0]),
                              (dets[i, :, 3] + dis[:, 1])]).T)
    dets_next = np.concatenate(dets_next, axis=0).reshape(-1,500,4)

    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], opt.num_classes, dets_next, speed)

    for j in range(1, opt.num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 10)

    return dets[0]


def merge_outputs(detections, opt):
    results = {}
    for j in range(1, opt.num_classes + 1):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0).astype(np.float32)

    scores = np.hstack(
        [results[j][:, 4] for j in range(1, opt.num_classes + 1)])
    if len(scores) > opt.K:
        kth = len(scores) - opt.K
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, opt.num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    return results


def letterbox(img, height=544, width=960,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh


def run_each_dataset(model_dir, model, dataset_path, subset, cur_dataset, opt):
    #print(cur_dataset)

    img_list = os.listdir(os.path.join(dataset_path, subset, cur_dataset))
    img_list = [os.path.join(dataset_path, subset, cur_dataset, _) for _ in img_list if ('jpg' in _) or ('png' in _)]
    img_list = sorted(img_list)
    img_len = len(img_list)
    last_feat = None

    confidence_threshold = 0.4
    IOU_threshold = 0.05
    retention_threshold = 10
    ret_threshold = 0.4

    det_list_all = []
    tracklet_all = []
    max_id = 0
    max_draw_len = 100
    draw_interval = 5
    img_width = 960
    img_height = 540
    fps = 25
    hm = torch.zeros(0)
    time_sum = 0

    for i in range(img_len):
        det_list_all.append([])

    for idx in range(img_len + 1):
        i = idx - 1
        # print('tracking: ', i)
        data_path1 = img_list[min(idx, img_len - 1)]
        # print(data_path1)
        img0 = cv2.imread(data_path1)  # BGR
        height, width, chance = img0.shape
        assert img0 is not None, 'Failed to load ' + data_path1

        # Padded resize
        # 可能需要调整输入resize大小
        img, ratio, padw, padh = letterbox(img0, height=544, width=960)
        img = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB

        # show
        # cv2.namedWindow("Image")
        # cv2.imshow("Image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows

        # Normalize RGB
        img = F.to_tensor(img).unsqueeze(0)  # .permute(0, 1, 3, 2)
        _, inp_chance, inp_height, inp_width = img.shape

        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // opt.down_ratio,
                'out_width': inp_width // opt.down_ratio}
        start1 = time.time()
        with torch.no_grad():
            # print(hm)

            output = model(img.cuda(), last_feat=last_feat, hm_pre=hm.cuda())[-1]
            last_feat = output['featmap'].cuda()
            hm = output['hm'].sigmoid_()

            if idx > 0:

                wh = output['wh']
                reg = None
                # id = output['id'].sigmoid_()
                # print(id.shape)
                dis = output['dis']
                speed = output['speed']
                # print(dis.shape )
                dets, inds = mot_decode(hm, wh, reg=reg, ltrb=opt.ltrb, K=opt.K)

                dis = _tranpose_and_gather_feat(dis, inds)
                dis = dis.squeeze(0)
                dis = dis.cpu().numpy()
                speed = _tranpose_and_gather_feat(speed, inds)
                speed = speed.squeeze(0)
                speed = speed.cpu().numpy()

                #根据clase拆分
                dets = post_process(dets, meta, opt, dis, speed)

                #取出全部类
                multi_classes_dets = merge_outputs([dets], opt)

                dets_list = []
                for key,value in multi_classes_dets.items():
                    remain_inds = value[:, 4] > opt.conf_thres
                    app_det = value[remain_inds, :]
                    if app_det.shape[0] == 0:
                        continue
                    app_det[:,4] = key
                    dets_list.append(app_det)
                    #print(key)
                if len(dets_list) > 0:
                    dets_list = np.concatenate([dets_list[i] for i in range(len(dets_list))], axis=0)
                else:
                    continue



                if dets_list.shape[0] > 0:
                    '''Detections'''
                    for j in range(dets_list.shape[0]):
                        x1 = int(dets_list[j, 0])
                        y1 = int(dets_list[j, 1])
                        x2 = int(dets_list[j, 2])
                        y2 = int(dets_list[j, 3])
                        x3 = int(dets_list[j, 5])
                        y3 = int(dets_list[j, 6])
                        x4 = int(dets_list[j, 7])
                        y4 = int(dets_list[j, 8])

                        det_rect = detect_rect()
                        det_rect.curr_frame = idx
                        det_rect.curr_rect = np.array([x1, y1, x2, y2])
                        det_rect.next_rect = np.array([x3, y3, x4, y4])
                        det_rect.type = int(dets_list[j, 4])
                        det_rect.speed = float(dets_list[j, 9])
                        det_list_all[det_rect.curr_frame - 1].append(det_rect)

                    if i == 0:
                        for j in range(len(det_list_all[i])):
                            det_list_all[i][j].id = j + 1
                            max_id = max(max_id, j + 1)
                            track = tracklet(det_list_all[i][j])
                            tracklet_all.append(track)
                        continue

                    matches, unmatched1, unmatched2 = track_det_match(tracklet_all, det_list_all[i], IOU_threshold)

                    for j in range(len(matches)):
                        det_list_all[i][matches[j][1]].id = tracklet_all[matches[j][0]].id
                        tracklet_all[matches[j][0]].add_rect(det_list_all[i][matches[j][1]])

                    delete_track_list = []
                    for j in range(len(unmatched1)):
                        # print("unmatched1")
                        tracklet_all[unmatched1[j]].no_match_frame = tracklet_all[unmatched1[j]].no_match_frame + 1
                        if (tracklet_all[unmatched1[j]].no_match_frame >= retention_threshold):
                            delete_track_list.append(unmatched1[j])

                    origin_index = set([k for k in range(len(tracklet_all))])
                    delete_index = set(delete_track_list)
                    left_index = list(origin_index - delete_index)
                    tracklet_all = [tracklet_all[k] for k in left_index]

                    for j in range(len(unmatched2)):
                        # print("unmatched2")
                        det_list_all[i][unmatched2[j]].id = max_id + 1
                        max_id = max_id + 1
                        track = tracklet(det_list_all[i][unmatched2[j]])
                        tracklet_all.append(track)
        time_sum = time_sum + (time.time() - start1)
    print(time_sum)

    # **************visualize tracking result and save evaluate file****************

    fout_tracking = open(os.path.join(model_dir, 'results', cur_dataset + '.txt'), 'w')

    save_img_dir = os.path.join(model_dir, 'results', cur_dataset)
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)

    out_video = os.path.join(model_dir, 'results', cur_dataset + '.mp4')
    videoWriter = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (img_width, img_height))

    id_dict = {}
    type_dict = {1:"Bus",2:"Car",3:"Van",4:"Others"}

    for i in range(img_len):
        # print('saving: ', i)
        img = cv2.imread(img_list[i])

        for j in range(len(det_list_all[i])):

            x1, y1, x2, y2 = det_list_all[i][j].curr_rect.astype(int)
            trace_id = det_list_all[i][j].id
            veh_type = det_list_all[i][j].type
            veh_speed = det_list_all[i][j].speed


            id_dict.setdefault(str(trace_id), []).append((int((x1 + x2) / 2), y2))
            draw_trace_id = str(trace_id)
            draw_caption(img, (x1, y1, x2, y2), draw_trace_id, color=color_list[trace_id % len(color_list)])
            cv2.rectangle(img, (x1, y1), (x2, y2), color=color_list[trace_id % len(color_list)], thickness=2)
            cv2.putText(img, str(type_dict[veh_type]), (x2+1, y1+11), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        color=color_list[trace_id % len(color_list)], thickness=1)
            cv2.putText(img, str(format(veh_speed,'.3f'))+"m/s", (x2+1, y1+21), cv2.FONT_HERSHEY_SIMPLEX,0.4,
                        color=color_list[trace_id % len(color_list)], thickness=1)
            trace_len = len(id_dict[str(trace_id)])
            trace_len_draw = min(max_draw_len, trace_len)

            for k in range(trace_len_draw - draw_interval):
                if (k % draw_interval == 0):
                    draw_point1 = id_dict[str(trace_id)][trace_len - k - 1]
                    draw_point2 = id_dict[str(trace_id)][trace_len - k - 1 - draw_interval]
                    cv2.line(img, draw_point1, draw_point2, color=color_list[trace_id % len(color_list)], thickness=2)

            fout_tracking.write(
                str(i + 1) + ',' + str(trace_id) + ',' + str(x1) + ',' + str(y1) + ',' + str(x2 - x1) + ',' + str(
                    y2 - y1) + ','+ str(veh_type)+ ','+ str(format(veh_speed,'.4f'))+',-1,-1\n')

        cv2.imwrite(os.path.join(save_img_dir, str(i + 1).zfill(6) + '.jpg'), img)
        videoWriter.write(img)
        cv2.waitKey(0)

    fout_tracking.close()
    videoWriter.release()


def run_from_train(model_dir, root_path):
    if not os.path.exists(os.path.join(model_dir, 'results')):
        os.makedirs(os.path.join(model_dir, 'results'))
    retinanet = torch.load(os.path.join(model_dir, 'model_final.pt'))

    use_gpu = True

    if use_gpu: retinanet = retinanet.cuda()

    retinanet.eval()

    for seq_num in [2, 4, 5, 9, 10, 11, 13]:
        run_each_dataset(model_dir, retinanet, root_path, 'train', 'MOT17-{:02d}'.format(seq_num))
    for seq_num in [1, 3, 6, 7, 8, 12, 14]:
        run_each_dataset(model_dir, retinanet, root_path, 'test', 'MOT17-{:02d}'.format(seq_num))


def main(args=None):
    opt = opts().parse()
    parser = argparse.ArgumentParser(description='Simple script for testing a CTracker network.')
    parser.add_argument('--dataset_path', default='./data/', type=str,
                        help='Dataset path, location of the images sequence.')
    parser.add_argument('--model_dir', default='./ctracker/', help='Path to model (.pt) file.')

    parser = parser.parse_args(args)
    model_heads = {'hm': 4, 'wh': 4, 'dis': 2, 'speed': 1}  # 'id': 1,
    if not os.path.exists(os.path.join(parser.model_dir, 'results')):
        os.makedirs(os.path.join(parser.model_dir, 'results'))

    model = get_pose_net(num_layers=34, heads=model_heads)
    model = load_model(model, os.path.join(parser.model_dir, 'model_30_6.pth'))
    # model.load(os.path.join(parser.model_dir, 'trackv0_model_last.pth'))
    print(model)
    # print(retinanet)

    use_gpu = True

    if use_gpu: model = model.cuda()

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    model.eval()

    # for seq_num in [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
    # run_each_dataset(parser.model_dir, model, parser.dataset_path, 'train', 'MOT15-{:02d}'.format(seq_num),opt)

    # for seq_num in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
    # run_each_dataset(parser.model_dir, model, parser.dataset_path, 'test', 'MOT15-{:02d}'.format(seq_num),opt)

    # for seq_num in [2, 4, 5, 9, 10, 11, 13]:
    # run_each_dataset(parser.model_dir, model, parser.dataset_path, 'train', 'MOT16-{:02d}'.format(seq_num),opt)

    for seq_num in [39031, 39051, 39211, 39271, 39311, 39361, 39371, 39401, 39501, 39511, 40701, 40711, 40712, 40714, 40742, 40743, 40761,
                    40762, 40763, 40771, 40772, 40773, 40774, 40775, 40792, 40793, 40851, 40852, 40853, 40854, 40855, 40863, 40864, 40891,
                    40892, 40901, 40902, 40903, 40904, 40905]:
        run_each_dataset(parser.model_dir, model, parser.dataset_path, 'Insight-MVT_Annotation_Test',
                         'MVI_{:05d}'.format(seq_num), opt)


if __name__ == '__main__':
    main()
