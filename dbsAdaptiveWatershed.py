#!/usr/bin/env python
# encoding: utf-8
'''
@author: du jianjun
@license: (C) Copyright 2019, PERSUPER, NERCITA.
@contact: dujianjun18@126.com
@software: wspp
@file: dbsAdaptiveWatershed.py
@time: 2021-06-04 14:27
@desc:

Evaluating multiple regions with the adaptive watershed-based approach.

'''
#########watershed#########
from skimage.feature import peak_local_max
# from skimage.morphology import watershed # 更新了！！！
from skimage.segmentation import watershed
from skimage import morphology
from scipy import ndimage
import numpy as np
import argparse
import imutils
import cv2
#########watershed#########

from dbsFunction import *

from collections import deque


class dbsAdaptiveWatershed():
    index_id = 0

    def __init__(self):
        pass

    def adaptiveWatershed(rgbImage, binImage, minArea=20, ratio_area_threshold=0.9, bGenerateVisImage = False):
        '''

        :param minArea:
        :param ratio_area_threshold: 实际面积/凸包面积，评估是否需要分裂的阈值，大于该值认为不用分裂
        :return:
        '''
        start = time.perf_counter()
        if cv2.__version__ <= '3.4.4':
            image0, contours, hierarchy_vbs = cv2.findContours(binImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours, hierarchy_vbs = cv2.findContours(binImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        rgbVis_split = cv2.cvtColor(binImage, cv2.COLOR_GRAY2BGR)  # dbsimread(fn, cv2.IMREAD_COLOR)
        mask_split = np.zeros(binImage.shape, np.uint8)

        contours_individualobject = []
        contours_multipleobject = []
        list_min_distance = []  #
        list_contours_splittedobjects = []
        list_offset = []
        list_cons_discard = []

        for id, con in enumerate(contours):
            area_con = cv2.contourArea(con)
            if area_con <= 100:
                contours_multipleobject.append(con)
                continue
            hull = cv2.convexHull(con)
            area_hull = cv2.contourArea(hull)
            if area_con < minArea:
                continue

            # print("ID:{}, con size:{} area_hull:{} area_con:{} scale:{}".format(id, len(con), area_hull, area_con, area_con / area_hull))
            if area_con / area_hull > ratio_area_threshold:
                contours_individualobject.append(con)
                cv2.drawContours(mask_split, [con], -1, 255, -1)
                cv2.drawContours(rgbVis_split, [con], -1, dbsFunction.getColor(dbsAdaptiveWatershed.index_id), 2)
                dbsAdaptiveWatershed.index_id += 1
            else:
                contours_multipleobject.append(con)
                # 第一种方法：同时分裂区域，难以确定参数min_distance
                # contours_splitted, offset, min_distance = dbsAdaptiveWatershed.splitIndividualConnectObject(con, minArea=20)
                # 第二种方法：依次提取区域，每个min_distance都是自适应的
                contours_splitted, cons_discard, offset, list_min_distance2 = \
                    dbsAdaptiveWatershed.adaptiveWatershed_IndividualObject(con, minArea,
                                                                            ratio_area_threshold=ratio_area_threshold)

                if len(cons_discard):
                    list_cons_discard.append(cons_discard)

                if len(contours_splitted) > 0:
                    list_contours_splittedobjects.append(contours_splitted)
                    list_offset.append(offset)

                    list_min_distance.append(list_min_distance2)

                    cv2.drawContours(mask_split, contours_splitted, -1, 255, -1, offset=(offset[0], offset[1]))
                    cv2.drawContours(mask_split, contours_splitted, -1, 0, 1, offset=(offset[0], offset[1]))

                    for one in contours_splitted:
                        cv2.drawContours(rgbVis_split, [one], -1, dbsFunction.getColor(dbsAdaptiveWatershed.index_id),
                                         -1, offset=(offset[0], offset[1]))
                        dbsAdaptiveWatershed.index_id += 1

        #
        mask_contours_splittedobjects = cv2.cvtColor(binImage, cv2.COLOR_GRAY2BGR)
        numberSplit = 0
        for idr, contours_res in enumerate(list_contours_splittedobjects):
            cv2.drawContours(mask_split, contours_res, -1, 0, 1)
            numberSplit += len(contours_res)

            offset = list_offset[idr]
            cv2.drawContours(mask_contours_splittedobjects, contours_res, -1, dbsFunction.getColor(idr), -1,
                             offset=offset)

            if rgbImage is not None:
                cv2.drawContours(rgbImage, contours_res, -1, dbsFunction.getColor(idr), 2,
                             offset=offset)
        if cv2.__version__ <= '3.4.4':
            image0, contours_results2, hierarchy_vbs = cv2.findContours(mask_split, cv2.RETR_TREE,
                                                                        cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours_results2, hierarchy_vbs = cv2.findContours(mask_split, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # -------------------------------------------------------------------------------
        tips = []
        tips.append("Segmented regions: {}".format(len(contours)))
        tips.append("Individual regions:{}".format(len(contours_individualobject)))
        tips.append("Candidate regions: {}".format(len(contours_multipleobject)))
        tips.append("Splitted regions:  {}".format(numberSplit))
        tips.append("Vascular bundles:  {}".format(len(contours_individualobject) + numberSplit))

        contours_results = []  # total
        contours_results.extend(contours_individualobject)
        for idr, contours_res in enumerate(list_contours_splittedobjects):
            contours_results.extend(contours_res)

        dict_cons = {}
        dict_cons["cons_ini"] = contours
        dict_cons["cons_individual"] = contours_individualobject
        dict_cons["cons_multiple"] = contours_multipleobject
        dict_cons["list_cons_splitted"] = list_contours_splittedobjects
        dict_cons["cons_results"] = contours_results
        for idt, tip in enumerate(tips):
            cv2.putText(mask_contours_splittedobjects, tips[idt], (20, idt * 60 + 80), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (255, 255, 255), 4)

            if rgbImage is not None:
                cv2.putText(rgbImage, tips[idt], (20, idt * 60 + 80), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (255, 255, 255), 4)

        #---------------------------------------------------------------------------------------
        # 20210623 下面功能只能用于单株尺度上不同节间的图像拼接
        vis_rgb_all = None
        if bGenerateVisImage:
            width = 1000
            visrgb_individual = dbsAdaptiveWatershed.stitchListofContours2Image(width, contours_individualobject)

            visrgb_multiple = dbsAdaptiveWatershed.stitchListofContours2Image(width, contours_multipleobject)

            list_masks = []
            for idd, cons in enumerate(list_contours_splittedobjects):
                con_before = contours_multipleobject[idd]

                x, y, w, h = cv2.boundingRect(con_before)

                # print("conbefor: {}, {}".format(w, h))
                mask_con = np.zeros((h, w, 3), np.uint8)
                cv2.drawContours(mask_con, [con_before], -1, (255, 255, 255), -1)
                for iddd, con in enumerate(cons):
                    cv2.drawContours(mask_con, [con], -1, dbsFunction.getColor(iddd), -1)

                #####
                # tag = list_tags[idd]
                # cv2.putText(mask_con, tag, (5, h+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

                list_masks.append(mask_con)

            list_tags = []
            for tags in list_min_distance:
                valtag = ""
                for i, val in enumerate(tags):
                    valtag += "{}".format(val)
                    if i != len(tags) - 1:
                        valtag += ":"

                # tag = "{}".format(tags)
                list_tags.append(valtag)
            visrgb_splitted = dbsAdaptiveWatershed.stitchListofImages2Image(width, list_masks, list_tags)

            list_visrgb = []
            list_visrgb.append(visrgb_individual)
            list_visrgb.append(visrgb_multiple)
            list_visrgb.append(visrgb_splitted)
            vis_rgb_all = dbsAdaptiveWatershed.stitchListofImagesEuqalWidth(width, list_visrgb)

            list_visrgb2 = []
            list_visrgb2.append(rgbImage)

            list_visrgb2.append(vis_rgb_all)
            vis_rgb_all = dbsAdaptiveWatershed.stitchListofImagesEuqalHeight(2000, list_visrgb2)

        return dict_cons, list_cons_discard, mask_split, vis_rgb_all

    def stitchListofContours2Image(width, cons, list_tags=[]):
        '''
        :param cons:
        :return:
        '''

        if len(cons) <= 0:
            visrgb_IO = np.zeros((100, width, 3), np.uint8)
            cv2.putText(visrgb_IO, "No candidate region to divide", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (255, 255, 255), 4)
            return visrgb_IO

        max_wid_IO = 0
        max_hei_IO = 0
        for con in cons:
            # print("con size:", len(con))
            x, y, cols, rows = cv2.boundingRect(con)
            if cols > max_wid_IO: max_wid_IO = cols
            if rows > max_hei_IO: max_hei_IO = rows

        num = len(cons)
        num_col = int(width / max_wid_IO)

        num_row = int(num / num_col) if int(num % num_col) == 0 else int(num / num_col) + 1
        height = int(num_row * max_hei_IO)

        visrgb_IO = np.zeros((height, width, 3), np.uint8)
        for idd, con in enumerate(cons):
            x, y, wid, hei = cv2.boundingRect(con)

            rows_id = int(idd / num_col)
            cols_id = int(idd % num_col)

            xx = cols_id * max_wid_IO - x
            yy = rows_id * max_hei_IO - y
            xx += int(max_wid_IO / 2 - wid / 2)
            yy += int(max_hei_IO / 2 - hei / 2)

            xx_l = cols_id * max_wid_IO
            yy_l = rows_id * max_hei_IO
            cv2.drawContours(visrgb_IO, [con], -1, (255, 255, 255), -1, offset=(xx, yy))

            if idd < len(list_tags):
                tag = list_tags[idd]
                cv2.putText(visrgb_IO, tag, (xx_l, yy_l + hei - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.rectangle(visrgb_IO, (0, 0), (width, height), (255, 255, 255), 3)

        return visrgb_IO

    def stitchListofImages2Image(width, list_images, list_tags=[]):
        '''
        :param cons:
        :return:
        '''
        if len(list_images) <= 0:
            visrgb_IO = np.zeros((100, width, 3), np.uint8)
            cv2.putText(visrgb_IO, "No candidate region to parse", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (255, 255, 255), 4)
            return visrgb_IO

        max_wid_IO = 0
        max_hei_IO = 0
        for sub in list_images:
            hei, wid = sub.shape[:2]
            if wid > max_wid_IO: max_wid_IO = wid
            if hei > max_hei_IO: max_hei_IO = hei

        num = len(list_images)
        num_col = int(width / max_wid_IO)

        hei_tag = 20

        num_row = int(num / num_col) if int(num % num_col) == 0 else int(num / num_col) + 1
        height = int(num_row * max_hei_IO) + hei_tag * num_row
        visrgb_IO = np.zeros((height, width, 3), np.uint8)

        for idd, sub in enumerate(list_images):
            hei, wid = sub.shape[:2]

            rows_id = int(idd / num_col)
            cols_id = int(idd % num_col)

            xx = cols_id * max_wid_IO
            yy = rows_id * max_hei_IO + rows_id * hei_tag
            xx2 = xx + int(max_wid_IO / 2 - wid / 2)
            yy2 = yy + int(max_hei_IO / 2 - hei / 2)

            xx_l = cols_id * max_wid_IO
            yy_l = rows_id * max_hei_IO

            visrgb_IO[yy2:yy2 + hei, xx2:xx2 + wid] = sub

            if idd < len(list_tags):
                tag = list_tags[idd]
                cv2.rectangle(visrgb_IO, (xx + 2, yy + max_hei_IO - 2),
                              (xx + max_wid_IO - 2, yy + max_hei_IO + hei_tag - 2), (128, 128, 128), -1)
                cv2.putText(visrgb_IO, tag, (xx + 3, yy + max_hei_IO + hei_tag - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

        cv2.rectangle(visrgb_IO, (0, 0), (width, height), (255, 255, 255), 3)

        return visrgb_IO

    def stitchListofImagesEuqalWidth(width, list_images):
        '''
        沿单行排列，图像等比调整为等高度
        :param cons:
        :return:
        '''
        list_images_new = []
        height = 0
        for sub in list_images:
            hei, wid = sub.shape[:2]
            scale = width / wid
            hei_new = int(hei * scale)
            sub_new = cv2.resize(sub, (width, hei_new))
            list_images_new.append(sub_new)
            height += hei_new

        visrgb_IO = np.zeros((height, width, 3), np.uint8)
        yy_new = 0
        for idd, sub in enumerate(list_images_new):
            hei, wid = sub.shape[:2]

            visrgb_IO[yy_new:yy_new + hei, 0:width] = sub
            yy_new += hei

        cv2.rectangle(visrgb_IO, (0, 0), (width, height), (255, 255, 255), 3)

        return visrgb_IO

    def stitchListofImagesEuqalHeight(height, list_images):
        '''
        沿单行排列，图像等比调整为等高度
        :param cons:
        :return:
        '''
        list_images_new = []
        width = 0
        for sub in list_images:
            # print("sub shape:", sub.shape)
            hei, wid = sub.shape[:2]
            scale = height / hei
            wid_new = int(wid * scale)
            sub_new = cv2.resize(sub, (wid_new, height))

            # print("sub_new shape:", sub_new.shape)
            list_images_new.append(sub_new)
            width += wid_new

        visrgb_IO = np.zeros((height, width, 3), np.uint8)
        # print("visrgb_IO shape:", visrgb_IO.shape)

        xx_new = 0
        for idd, sub in enumerate(list_images_new):
            hei, wid = sub.shape[:2]

            visrgb_IO[0:height, xx_new:xx_new + wid] = sub
            xx_new += wid
        cv2.rectangle(visrgb_IO, (0, 0), (width, height), (255, 255, 255), 3)
        return visrgb_IO

    def splitIndividualConnectObject(contour, minArea=20):
        #
        # 注意：在原始分割图像上进行分解：
        # 20210604
        x, y, cols, rows = cv2.boundingRect(contour)
        offset = (x, y)

        mask_watershed = np.zeros((rows, cols), np.uint8)
        cv2.drawContours(mask_watershed, [contour], -1, 255, -1, offset=(-x, -y))

        """
        Get the maximum/largest inscribed circle inside mask/polygon/contours.
        Support non-convex/hollow shape
        """
        dist_map = cv2.distanceTransform(mask_watershed, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        _, radius, _, center = cv2.minMaxLoc(dist_map)
        contours, _ = cv2.findContours(mask_watershed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        center2, radius2 = cv2.minEnclosingCircle(np.concatenate(contours, 0))
                #
        min_distance = int(radius / 2)
        # compute the exact Euclidean distance from every binary
        # pixel to the nearest zero pixel, then find peaks in this
        # distance map
        D = ndimage.distance_transform_edt(mask_watershed)
        localMax = peak_local_max(D,
                                  indices=False,
                                  min_distance=min_distance,
                                  # find peaks (i.e., local maxima) in the map. We’ll ensure that is at least a 20 pixel distance between each peak.
                                  labels=mask_watershed,
                                  exclude_border=True)
        # print("local max:", localMax): 标记矩阵
        # perform a connected component analysis on the local peaks,
        # using 8-connectivity, then appy the Watershed algorithm
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers,
                           mask=mask_watershed)  # Each pixel value as a unique label value. Pixels that have the same label value belong to the same object
        # print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
        # The last step is to simply loop over the unique label values and extract each of the unique objects:
        # loop over the unique labels returned by the Watershed
        # algorithm
        contours_splitted = []
        for label in np.unique(labels):
            # if the label is zero, we are examining the 'background'
            # so simply ignore it
            if label == 0:
                continue
            # otherwise, allocate memory for the label region and draw
            # it on the mask
            mask__ = np.zeros(mask_watershed.shape, dtype="uint8")
            mask__[labels == label] = 255
            # detect contours in the mask and grab the largest one
            cnts = cv2.findContours(mask__.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)
            # draw a circle enclosing the object
            ((x, y), r) = cv2.minEnclosingCircle(c)
            # cv2.circle(rgbVis_split, (int(x), int(y)), int(r), dbsFunction.getColor(label),1)
            # (0, 255, 0), 1)

            ar = cv2.contourArea(c)
            if ar > minArea:
                # cv2.drawContours(visImage, [c], -1, dbsFunction.getColor(dbsAdaptiveWatershed.index_id), 2)
                dbsAdaptiveWatershed.index_id += 1
                contours_splitted.append(c)

        return contours_splitted, offset, min_distance

    def adaptiveWatershed_IndividualObject(contour, minArea=20, ratio_area_threshold=0.9):
        '''
        :param minArea:
        :return: 返回最大轮廓和偏移
        '''

        x, y, cols, rows = cv2.boundingRect(contour)
        offset = (x, y)

        mask_one = np.zeros((rows, cols), np.uint8)
        cv2.drawContours(mask_one, [contour], -1, 255, -1, offset=(-x, -y))
        contours, _ = cv2.findContours(mask_one, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        cons_discard = []  #

        cons_outs = []
        list_min_distance = []

        dq_todo = deque()
        for con in contours:
            a = cv2.contourArea(con)
            if a > minArea:
                dq_todo.append(con.copy())

        while len(dq_todo)>0:
            cont_one = dq_todo.popleft()

            hull = cv2.convexHull(cont_one)
            area_hull = cv2.contourArea(hull)
            area_con = cv2.contourArea(cont_one)

            if area_con < minArea:
                cons_discard.append(cont_one)
                continue

            if area_con / area_hull > ratio_area_threshold:
                cons_outs.append(cont_one)
                continue

            mask_one = np.zeros((rows, cols), np.uint8)
            cv2.drawContours(mask_one, [cont_one], -1, 255, -1)  # , offset=(-x,-y))
            dist_map = cv2.distanceTransform(mask_one, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            _, radius, _, center = cv2.minMaxLoc(dist_map)
            if radius < 2:
                cons_discard.append(cont_one)
                continue

            # 计算最小外接圆
            contours, _ = cv2.findContours(mask_one, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            center2, radius2 = cv2.minEnclosingCircle(np.concatenate(contours, 0))  # ?
            ptMinCir = (int(center2[0]), int(center2[1]))

            # 取最大内切圆半径最为两个目标之间最小距离参数
            min_distance = int(radius * 0.6)  # * 1.5) #2
            if min_distance < 2:
                cons_discard.append(cont_one)
                continue

            list_min_distance.append(min_distance)

            # compute the exact Euclidean distance from every binary
            # pixel to the nearest zero pixel, then find peaks in this
            # distance map
            D = ndimage.distance_transform_edt(mask_one)
            localMax = peak_local_max(D,
                                      indices=False,
                                      min_distance=min_distance,
                                      # find peaks (i.e., local maxima) in the map. We’ll ensure that is at least a 20 pixel distance between each peak.
                                      labels=mask_one,
                                      exclude_border=True)
            # print("local max:", localMax): 标记矩阵
            # perform a connected component analysis on the local peaks,
            # using 8-connectivity, then appy the Watershed algorithm
            markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
            labels = watershed(-D, markers,
                               mask=mask_one)  # Each pixel value as a unique label value. Pixels that have the same label value belong to the same object
            # print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

            # The last step is to simply loop over the unique label values and extract each of the unique objects:
            # loop over the unique labels returned by the Watershed
            # algorithm
            c_s_res = []
            for label in np.unique(labels):
                # if the label is zero, we are examining the 'background'
                # so simply ignore it
                if label == 0:
                    continue
                # otherwise, allocate memory for the label region and draw
                # it on the mask
                mask__ = np.zeros(mask_one.shape, dtype="uint8")
                mask__[labels == label] = 255
                # detect contours in the mask and grab the largest one
                cnts = cv2.findContours(mask__.copy(), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                c = max(cnts, key=cv2.contourArea)
                # draw a circle enclosing the object
                # ((x, y), r) = cv2.minEnclosingCircle(c)
                # cv2.circle(rgbVis_split, (int(x), int(y)), int(r), dbsFunction.getColor(label),1)
                # (0, 255, 0), 1)

                c_s_res.append(c)

            if len(c_s_res) <= 0:
                cons_discard.append(cont_one)
                continue

            c_max = []
            a_max = 0
            for c in c_s_res:
                a = cv2.contourArea(c)
                if a > a_max:
                    a_max = a
                    c_max = c
            # print("max area:", a_max, "  c_s_res:", len(c_s_res))
            if a_max > minArea:
                cons_outs.append(c_max)
                cv2.drawContours(mask_one, [c_max], -1, 0, -1)


            contours = dbsFunction.GetOuterContours(mask_one)
            for con in contours:
                a = cv2.contourArea(con)
                if a > minArea:
                    dq_todo.append(con.copy())
                else:
                    cons_discard.append(cont_one)

        return cons_outs, cons_discard, offset, list_min_distance


if __name__ == "__main__":
    exit(0)


