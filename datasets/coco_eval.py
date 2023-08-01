import os
import contextlib
import copy
import numpy as np

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

from utils.box_utils import convert_to_xywh
from utils.distributed_utils import all_gather


class CocoEval(COCOeval):

    def __init__(self, coco_gt=None, coco_dt=None, iou_type='bbox'):
        super(CocoEval, self).__init__(coco_gt, coco_dt, iou_type)

    def evaluate(self):
        p = self.params
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p
        self._prepare()
        cat_ids = p.catIds if p.useCats else [-1]
        self.ious = {
            (imgId, catId): self.computeIoU(imgId, catId)
            for imgId in p.imgIds
            for catId in cat_ids
        }
        eval_imgs = [
            self.evaluateImg(imgId, catId, areaRng, p.maxDets[-1])
            for catId in cat_ids
            for areaRng in p.areaRng
            for imgId in p.imgIds
        ]
        eval_imgs = np.asarray(eval_imgs).reshape(len(cat_ids), len(p.areaRng), len(p.imgIds))
        self._paramsEval = copy.deepcopy(self.params)
        return p.imgIds, eval_imgs

    def summarize_ap(self, if_print=True):

        def _summarize(iou_thr=None, area_rng='all', max_dets=100):
            p = self.params
            iou_str = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iou_thr is None else '{:0.2f}'.format(iou_thr)
            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == area_rng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == max_dets]
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            if iou_thr is not None:
                t = np.where(iou_thr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]
            aps = np.asarray([np.mean(s[:, :, i, :]) for i in range(s.shape[2])])
            aps_clean = [ap for ap in aps if ap > -0.001]
            mean_ap = np.mean(aps_clean)
            if if_print:
                print('Mean Average Precision (mAP) @ [ IoU='
                      + iou_str + ' | area=' + area_rng + ' | max_dets=' + str(max_dets) + ' ] = ' + str(mean_ap))
                for i, ap in enumerate(aps):
                    print('\tAP of category [' + self.cocoGt.cats[i + 1]['name'] + ']:\t\t' + str(ap))
            return aps

        if not self.eval:
            raise Exception('Please run accumulate() first')
        return _summarize(iou_thr=0.5, max_dets=self.params.maxDets[2])


class CocoEvaluator:

    def __init__(self, coco_gt):
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        self.coco_eval = CocoEval(coco_gt)
        self.img_ids = []
        self.eval_imgs = []

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)
        results = self.prepare_for_coco_detection(predictions)
        # suppress pycocotools prints
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
        self.coco_eval.cocoDt = coco_dt
        self.coco_eval.params.imgIds = list(img_ids)
        img_ids, eval_imgs = self.coco_eval.evaluate()
        self.eval_imgs.append(eval_imgs)

    def synchronize_between_processes(self):
        self.eval_imgs = np.concatenate(self.eval_imgs, 2)
        img_ids, eval_imgs = self.merge(self.img_ids, self.eval_imgs)
        img_ids, eval_imgs = list(img_ids), list(eval_imgs.flatten())
        self.coco_eval.evalImgs = eval_imgs
        self.coco_eval.params.imgIds = img_ids
        self.coco_eval._paramsEval = copy.deepcopy(self.coco_eval.params)

    def accumulate(self):
        self.coco_eval.accumulate()

    def summarize(self, if_print=True):
        return self.coco_eval.summarize_ap(if_print)

    @staticmethod
    def prepare_for_coco_detection(predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue
            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            coco_results.extend([
                {"image_id": original_id, "category_id": labels[k], "bbox": box, "score": scores[k]}
                for k, box in enumerate(boxes)
            ])
        return coco_results

    @staticmethod
    def merge(img_ids, eval_imgs):
        all_img_ids = all_gather(img_ids)
        all_eval_imgs = all_gather(eval_imgs)
        merged_img_ids = []
        for p in all_img_ids:
            merged_img_ids.extend(p)
        merged_eval_imgs = [p for p in all_eval_imgs]
        merged_img_ids = np.array(merged_img_ids)
        merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)
        # keep only unique (and in sorted order) images
        merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
        merged_eval_imgs = merged_eval_imgs[..., idx]
        return merged_img_ids, merged_eval_imgs
