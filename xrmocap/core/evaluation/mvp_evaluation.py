# yapf: disable
import logging
import numpy as np
import os
import time
import torch
from prettytable import PrettyTable
from torch.utils.data import DataLoader
from typing import List, Tuple, Union
from xrprimer.utils.log_utils import get_logger

from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.model.architecture.base_architecture import BaseArchitecture
from xrmocap.utils.distribute_utils import collect_results, is_main_process
from xrmocap.utils.mvp_utils import (
    AverageMeter, convert_result_to_kps, norm2absolute,
)

# yapf: enable


class MVPEvaluation:
    """Evaluation for MvP method."""

    def __init__(
        self,
        test_loader: DataLoader,
        dataset_name: Union[None, str] = None,
        print_freq: int = 100,
        final_output_dir: Union[None, str] = None,
        logger: Union[None, str, logging.Logger] = None,
    ):
        """Initialization for the class.

        Args:
            test_loader (DataLoader):
                Test dataloader.
            dataset_name (Union[None, str], optional):
                Name of the dataset. Defaults to None.
            print_freq (int, optional):
                Printing frequency. Defaults to 100.
            final_output_dir (Union[None, str], optional):
                Directory to output folder. Defaults to None.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """

        self.logger = get_logger(logger)

        self.test_loader = test_loader
        self.dataset_name = dataset_name

        self.print_freq = print_freq
        self.output_dir = final_output_dir

        self.dataset = test_loader.dataset
        self.gt_num = test_loader.dataset.len
        self.n_views = test_loader.dataset.n_views

    def run(
        self,
        model: BaseArchitecture,
        threshold: float = 0.1,
        is_train: bool = False,
    ):

        # validate model
        preds_single, _ = self.model_validate(
            model,
            threshold=threshold,
            is_train=is_train,
        )
        preds = collect_results(preds_single, len(self.dataset))

        # quantitative evaluation and print result
        if is_main_process():
            precision = None
            if 'panoptic' in self.dataset_name:
                tb = PrettyTable()
                mpjpe_threshold = np.arange(25, 155, 25)

                aps, recs, mpjpe, recall500 = \
                    self.evaluate_map(preds)

                tb.field_names = ['Threshold/mm'] + \
                    [f'{i}' for i in mpjpe_threshold]
                tb.add_row(['AP'] + [f'{ap * 100:.2f}' for ap in aps])
                tb.add_row(['Recall'] + [f'{re * 100:.2f}' for re in recs])
                tb.add_row(['recall@500mm'] +
                           [f'{recall500 * 100:.2f}' for re in recs])
                self.logger.info('\n' + tb.get_string())
                self.logger.info(f'MPJPE: {mpjpe:.2f}mm')

                precision = np.mean(aps[0])

            elif 'campus' in self.dataset_name \
                    or 'shelf' in self.dataset_name:

                actor_pcp, avg_pcp, recall500 = self.evaluate_pcp(
                    preds, recall_threshold=500, alpha=0.5)

                tb = PrettyTable()
                tb.field_names = [
                    'Metric', 'Actor 1', 'Actor 2', 'Actor 3', 'Average'
                ]
                tb.add_row([
                    'PCP', f'{actor_pcp[0] * 100:.2f}',
                    f'{actor_pcp[1] * 100:.2f}', f'{actor_pcp[2] * 100:.2f}',
                    f'{avg_pcp * 100:.2f}'
                ])
                self.logger.info('\n' + tb.get_string())
                self.logger.info(f'Recall@500mm: {recall500:.4f}')

                precision = np.mean(avg_pcp)

            else:
                self.logger.warning(f'Dataset {self.dataset_name} '
                                    'is not yet implemented.')
                raise NotImplementedError

        return precision

    def model_validate(self,
                       model: BaseArchitecture,
                       threshold: float,
                       is_train: bool = False):
        """Evaluate model during training or testing.

        Args:
            model (BaseArchitecture):
                Model to be evaluated.
            threshold (float):
                Confidence threshold to filter non-human keypoints.
            is_train (bool, optional):
                True if it is called during trainig. Defaults to False.

        Returns:
            float: model evaluation result, precision
        """
        batch_time = AverageMeter()
        data_time = AverageMeter()

        model.eval()

        preds = []
        keypoints3d = None
        with torch.no_grad():
            end = time.time()
            kps3d_pred = []
            n_max_person = 0
            for i, (inputs, meta) in enumerate(self.test_loader):
                data_time.update(time.time() - end)
                assert len(inputs) == self.n_views
                output = model(views=inputs, meta=meta)

                gt_kps3d = meta[0]['kps3d'].float()
                n_kps = gt_kps3d.shape[2]
                bs, n_queries = output['pred_logits'].shape[:2]

                src_poses = output['pred_poses']['outputs_coord']. \
                    view(bs, n_queries, n_kps, 3)
                src_poses = norm2absolute(src_poses, model.module.grid_size,
                                          model.module.grid_center)
                score = output['pred_logits'][:, :, 1:2].sigmoid()
                score = score.unsqueeze(2).expand(-1, -1, n_kps, -1)
                temp = (score > threshold).float() - 1

                pred = torch.cat([src_poses, temp, score], dim=-1)
                pred = pred.detach().cpu().numpy()
                for b in range(pred.shape[0]):
                    preds.append(pred[b])

                batch_time.update(time.time() - end)
                end = time.time()
                if (i % self.print_freq == 0 or i
                        == len(self.test_loader) - 1) and is_main_process():
                    gpu_memory_usage = torch.cuda.memory_allocated(0)
                    speed = len(inputs) * inputs[0].size(0) / batch_time.val
                    msg = f'Test: [{i}/{len(self.test_loader)}]\t' \
                        f'Time: {batch_time.val:.3f}s ' \
                        f'({batch_time.avg:.3f}s)\t' \
                        f'Speed: {speed:.1f} samples/s\t' \
                        f'Data: {data_time.val:.3f}s ' \
                        f'({data_time.avg:.3f}s)\t' \
                        f'Memory {gpu_memory_usage:.1f}'
                    self.logger.info(msg)

                if not is_train:
                    n_person, per_frame_kps3d = convert_result_to_kps(pred)
                    n_max_person = max(n_person, n_max_person)
                    kps3d_pred.append(per_frame_kps3d)

            if not is_train:
                n_frame = len(kps3d_pred)
                n_kps = n_kps
                kps3d = np.full((n_frame, n_max_person, n_kps, 4), np.nan)

                for frame_idx in range(n_frame):
                    per_frame_kps3d = kps3d_pred[frame_idx]
                    n_person = len(per_frame_kps3d)
                    if n_person > 0:
                        kps3d[frame_idx, :n_person] = per_frame_kps3d

                keypoints3d = Keypoints(kps=kps3d, convention=None)
                kps3d_file = os.path.join(self.output_dir, 'kps3d.npz')
                if is_main_process():
                    self.logger.info(f'Saving 3D keypoints to: {kps3d_file}')
                keypoints3d.dump(kps3d_file)

        return preds, keypoints3d

    def evaluate_map(self, pred_kps3d: torch.Tensor, threshold: float = 0.1) \
            -> Tuple[List[float], List[float], float, float]:
        """Evaluate MPJPE, mAP and recall based on MPJPE. Mainly for panoptic
        predictions.

        Args:
            pred_kps3d (torch.Tensor):
                Predicted 3D keypoints.

        Returns:
            Tuple[List[float], List[float], float, float]:
                List of AP, list of recall, MPJPE value and recall@500mm.
        """

        eval_list = []
        assert len(pred_kps3d) == self.gt_num, \
            f'number mismatch {len(pred_kps3d)} pred and {self.gt_num} gt'

        trans_ground = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0],
                                     [0.0, 1.0, 0.0]]).double()

        total_gt = 0
        for index in range(self.gt_num):
            scene_idx, frame_idx, _ = self.dataset.process_index_mapping(index)
            gt_keypoints3d = self.dataset.gt3d[scene_idx]
            gt_scene_kps3d = gt_keypoints3d.get_keypoints(
            )  # [n_frame, n_person, n_kps, 4]
            gt_frame_kps3d = gt_scene_kps3d[
                frame_idx][:, :self.dataset.n_kps, :]  # [n_person, n_kps, 4]

            check_valid = torch.sum(gt_frame_kps3d, axis=1)  # [n_person, 4]
            gt_frame_kps3d = gt_frame_kps3d[check_valid[:, -1] > 0]

            gt_frame_kps3d = [
                torch.mm(gt_person_kps3d[:, 0:3], trans_ground)
                for gt_person_kps3d in gt_frame_kps3d
            ]
            if len(gt_frame_kps3d) == 0:
                continue

            pred_frame_kps3d = pred_kps3d[index].copy()
            pred_frame_kps3d_valid = pred_frame_kps3d[pred_frame_kps3d[:, 0,
                                                                       3] >= 0]

            for pred_person_kps3d in pred_frame_kps3d_valid:
                mpjpes = []
                for gt_person_kps3d in gt_frame_kps3d:
                    vis = gt_person_kps3d[:, -1] > threshold

                    mpjpe = np.mean(
                        np.sqrt(
                            np.sum(
                                (np.array(pred_person_kps3d[vis, 0:3]) -
                                 np.array(gt_person_kps3d[vis, 0:3]))**2,
                                axis=-1)))
                    mpjpes.append(mpjpe)
                min_gt = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                score = pred_person_kps3d[0, 4]
                eval_list.append({
                    'mpjpe': float(min_mpjpe),
                    'score': float(score),
                    'gt_id': int(total_gt + min_gt)
                })

            total_gt += len(gt_frame_kps3d)

        mpjpe_threshold = np.arange(25, 155, 25)
        aps = []
        recs = []
        for t in mpjpe_threshold:
            ap, rec = self._eval_list_to_ap(eval_list, total_gt, t)
            aps.append(ap)
            recs.append(rec)

        return \
            aps, \
            recs, \
            self._eval_list_to_mpjpe(eval_list), \
            self._eval_list_to_recall(eval_list, total_gt)

    def evaluate_pcp(self,
                     pred_kps3d: torch.Tensor,
                     recall_threshold: int = 500,
                     threshold: float = 0.1,
                     alpha: float = 0.5) -> Tuple[List[float], float, float]:
        """Evaluate MPJPE and PCP. Mainly for Shelf and Campus predictions.

        Args:
            pred_kps3d (torch.Tensor):
                Predicted 3D keypoints.
            recall_threshold (int, optional):
                Threshold for MPJPE. Defaults to 500.
            alpha (float, optional):
                Threshold for correct limb part. Defaults to 0.5.
                Predicted limb part is regarded as correct if
                predicted_part_length < alpha * gt_part_length
        Returns:
            Tuple[List[float], float, float]:
                List of PCP per actor, average PCP, and recall@500mm.
        """
        assert len(pred_kps3d) == self.gt_num, \
            f'number mismatch {len(pred_kps3d)} pred and {self.gt_num} gt'

        trans_ground = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                     [0.0, 0.0, 1.0]]).double()

        limbs = [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [9, 10],
                 [10, 11], [12, 13]]

        gt_n_person = np.max(
            np.array([
                scene_keypoints.get_keypoints().shape[1]
                for scene_keypoints in self.dataset.gt3d
            ]))

        correct_parts = np.zeros(gt_n_person)
        total_parts = np.zeros(gt_n_person)
        limb_correct_parts = np.zeros((gt_n_person, 10))

        total_gt = 0
        match_gt = 0
        for index in range(self.gt_num):
            scene_idx, frame_idx, _ = self.dataset.process_index_mapping(index)
            gt_keypoints3d = self.dataset.gt3d[scene_idx]
            gt_scene_kps3d = gt_keypoints3d.get_keypoints(
            )  # [n_frame, n_person, n_kps, 4]
            gt_frame_kps3d = gt_scene_kps3d[
                frame_idx][:, :self.dataset.n_kps, :]  # [n_person, n_kps, 4]

            gt_frame_kps3d = [
                torch.mm(gt_person_kps3d[:, 0:3], trans_ground)
                for gt_person_kps3d in gt_frame_kps3d
            ]
            if len(gt_frame_kps3d) == 0:
                continue

            pred_frame_kps3d = pred_kps3d[index].copy()
            pred_frame_kps3d_valid = pred_frame_kps3d[pred_frame_kps3d[:, 0, 3]
                                                      >= 0]  # if is a person

            for person_idx, gt_person_kps3d in enumerate(gt_frame_kps3d):

                vis = gt_person_kps3d[:, -1] > threshold

                check_valid = torch.sum(gt_person_kps3d, axis=1)  # [4]
                if check_valid[-1] == 0:
                    continue

                mpjpes = np.mean(
                    np.sqrt(
                        np.sum(
                            (np.array(pred_frame_kps3d_valid[:, vis, 0:3]) -
                             np.array(gt_person_kps3d[vis, 0:3]))**2,
                            axis=-1)),
                    axis=-1)
                min_n = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)

                if min_mpjpe < recall_threshold:
                    match_gt += 1

                total_gt += 1

                for j, k in enumerate(limbs):
                    total_parts[person_idx] += 1
                    error_s = \
                        np.linalg.norm(np.array(
                            pred_frame_kps3d_valid[min_n, k[0], 0:3]) -
                            np.array(gt_person_kps3d[k[0]]))
                    error_e = \
                        np.linalg.norm(np.array(
                            pred_frame_kps3d_valid[min_n, k[1], 0:3]) -
                            np.array(gt_person_kps3d[k[1]]))
                    limb_length = np.linalg.norm(
                        np.array(gt_person_kps3d[k[0]]) -
                        np.array(gt_person_kps3d[k[1]]))

                    if (error_s + error_e) / 2.0 <= alpha * limb_length:
                        correct_parts[person_idx] += 1
                        limb_correct_parts[person_idx, j] += 1

                pred_hip = (pred_frame_kps3d_valid[min_n, 2, 0:3] +
                            pred_frame_kps3d_valid[min_n, 3, 0:3]) / 2.0
                gt_hip = (gt_person_kps3d[2] + gt_person_kps3d[3]) / 2.0
                total_parts[person_idx] += 1
                error_s = np.linalg.norm(np.array(pred_hip) - np.array(gt_hip))
                error_e = np.linalg.norm(
                    np.array(pred_frame_kps3d_valid[min_n, 12, 0:3]) -
                    np.array(gt_person_kps3d[12]))
                limb_length = np.linalg.norm(
                    np.array(gt_hip) - np.array(gt_person_kps3d[12]))

                if (error_s + error_e) / 2.0 <= alpha * limb_length:
                    correct_parts[person_idx] += 1
                    limb_correct_parts[person_idx, 9] += 1

        actor_pcp = correct_parts / (total_parts + 1e-8)
        avg_pcp = np.mean(actor_pcp[:3])

        return \
            actor_pcp, avg_pcp, match_gt / (total_gt + 1e-8)

    def _eval_list_to_ap(self, eval_list, total_gt, threshold):
        """convert evaluation result to ap."""
        eval_list.sort(key=lambda k: k['score'], reverse=True)
        total_num = len(eval_list)

        tp = np.zeros(total_num)
        fp = np.zeros(total_num)
        gt_det = []
        for i, item in enumerate(eval_list):
            if item['mpjpe'] < threshold and item['gt_id'] not in gt_det:
                tp[i] = 1
                gt_det.append(item['gt_id'])
            else:
                fp[i] = 1
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / (total_gt + 1e-5)
        precise = tp / (tp + fp + 1e-5)
        for n in range(total_num - 2, -1, -1):
            precise[n] = max(precise[n], precise[n + 1])

        precise = np.concatenate(([0], precise, [0]))
        recall = np.concatenate(([0], recall, [1]))
        index = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[index + 1] - recall[index]) * precise[index + 1])

        return ap, recall[-2]

    def _eval_list_to_mpjpe(self, eval_list, threshold=500):
        """convert evaluation result to mpjpe."""
        eval_list.sort(key=lambda k: k['score'], reverse=True)
        gt_det = []

        mpjpes = []
        for i, item in enumerate(eval_list):
            if item['mpjpe'] < threshold and item['gt_id'] not in gt_det:
                mpjpes.append(item['mpjpe'])
                gt_det.append(item['gt_id'])

        return np.mean(mpjpes) if len(mpjpes) > 0 else np.inf

    def _eval_list_to_recall(self, eval_list, total_gt, threshold=500):
        """convert evaluation result to recall."""
        gt_ids = [e['gt_id'] for e in eval_list if e['mpjpe'] < threshold]

        return len(np.unique(gt_ids)) / total_gt
