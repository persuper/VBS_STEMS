#!/usr/bin/env python
# encoding: utf-8
'''
@author: du jianjun
@license: (C) Copyright 2019, PERSUPER, NERCITA.
@contact: dujianjun18@126.com
@software: wspp
@file: dbs_common_function.py
@time: 2020-01-07 23:33
@desc:
'''

import datetime
import os,time, re, math
import cv2
import csv
from PIL import Image
from PIL.ExifTags import TAGS
import exifread
import json
import urllib.request
import numpy as np
import matplotlib.pyplot as plt

class dbsFunction:
    def __init__(self):
        pass

    def distance(pt1=[], pt2=[]):
        return math.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))

    # 获得欧几里距离
    def _get_eucledian_distance(vect1, vect2):
        distant = vect1[0] - vect2[0]
        dist = np.sqrt(np.sum(np.square(distant)))
        # 或者用numpy内建方法
        # vect1 = list(vect1)
        # vect2 = list(vect2)
        # dist = np.linalg.norm(vect1 - vect2)
        return dist

    def line_intersection(line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])  # Typo was here

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception('lines do not intersect')

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y

    # 线段交点
    def findIntersection(x1, y1, x2, y2, x3, y3, x4, y4):
        if (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) == 0:
            px = 0
        else:
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
                        (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        if (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) == 0:
            py = 0
        else:
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
                    (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        return [px, py]

    g_colors = []
    def getColor(index):
        if len(dbsFunction.g_colors) <= 0:
            for i in range(1000):
                r = np.random.uniform(50, 255)
                g = np.random.uniform(50, 255)
                b = np.random.uniform(50, 255)
                dbsFunction.g_colors.append(
                    (r, g, b))

        index = index % 1000
        return dbsFunction.g_colors[index]

    def removedirfiles(path):
        import os
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    def get_FileCreateTime(filePath):
        # tm1 = time.ctime(os.stat(filePath.st_mtime)# 文件的修改时间
        tm2 = time.ctime(os.stat(filePath).st_ctime)  # 文件的创建时间
        print(tm2)

        return tm2

    def file_is_img(filename):
        ext = os.path.splitext(filename)[1]
        imgType_list = {'.JPG','.BMP','.TIF','.TIFF','.jpg', '.bmp', '.png', '.jpeg', '.rgb', '.tif'}
        if ext in imgType_list:
            return True
        else:
            return False

    def file_is_txt(filename):
        ext = os.path.splitext(filename)[1]
        Type_list = {'.txt','.TXT'}
        if ext in Type_list:
            return True
        else:
            return False

    def file_is_video(filename):
        ext = os.path.splitext(filename)[1]
        #print("ext:", ext)
        videoType_list = {'.mp4', '.mov', '.MP4', '.MOV'}
        if ext in videoType_list:
            return True
        else:
            return False
    def findfullimagefile(pathname):
        tfiles = os.listdir(pathname)

        fs = []
        for tfile in tfiles:
            if dbsFunction.file_is_img(tfile):
                fullfilename = os.path.join(pathname, tfile)
                fs.append(fullfilename)
        return fs


    def findfullTxtfile(pathname):
        tfiles = os.listdir(pathname)

        fs = []
        for tfile in tfiles:
            if dbsFunction.file_is_txt(tfile):
                fullfilename = os.path.join(pathname, tfile)
                fs.append(fullfilename)
        return fs

    def findfullvideofile(pathname):
        tfiles = os.listdir(pathname)
        fs = []
        for tfile in tfiles:
            filename = os.path.join(pathname, tfile)
            if dbsFunction.file_is_video(filename):
                fs.append(filename)
        return fs

    def getOnlyName(fullfilename):
        (root, filename) = os.path.split(fullfilename)
        (onlyname, extname) = os.path.splitext(filename)
        return onlyname

    def getDataDesp( inputstring, startid=0, endid=3):
        # _2020-01-09-15-19-31-675-144-0-100-0_
        print(inputstring)
        strOut = re.split('[_.-]', inputstring)  # 多个字符分割串
        strdata = "%.4d-%.2d-%.2d"%( int(strOut[0]), int(strOut[1]), int(strOut[2]))
        return strdata

    def findSubDirectory(pathname):
        out_subdirs = []

        names = os.listdir(pathname)
        for name in names:
            fulldir = os.path.join(pathname, name)
            if os.path.isdir(fulldir):
               out_subdirs.append(name)

        #for root, dirs, files in os.walk(pathname):
        #    for idir, dir in enumerate(dirs):
        #        fulldir = os.path.join(root, dir)
        #        if os.path.isdir(fulldir):
        #            out_subdirs.append(dir)
        return out_subdirs
    def findSubFullDirectory(pathname):
        out_subdirs = []

        names = os.listdir(pathname)
        for name in names:
            fulldir = os.path.join(pathname, name)
            if os.path.isdir(fulldir):
               out_subdirs.append(fulldir)
        return out_subdirs
    def parseString(strInput):
        # strOut = strInput.partition('_') # 只能找到第一个符号
        strOut = re.split('[_.-]', strInput)  # 多个字符分割串
        return strOut

    ##################################################
    # 读取中文名称的图像
    def dbsimread(image_path, flags=cv2.IMREAD_COLOR):
        original_image = cv2.imdecode(
            np.fromfile(image_path, dtype=np.uint8), flags)
        return original_image

    def dbsimwrite(image_path, original_image, ext='.jpg'):
        if ext == '.jpg':
            compress_rate = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
            cv2.imencode(ext, original_image, compress_rate)[1].tofile(image_path)
        else:
            cv2.imencode(ext, original_image)[1].tofile(image_path)

    def GetTotalArea(cons):
        totalarea = 0
        for i, cnt in enumerate(cons):
            totalarea += cv2.contourArea(cnt)
        return totalarea

    def GetLargestContour(mask, method=cv2.CHAIN_APPROX_SIMPLE):
        if cv2.__version__ <= '3.4.4':
            img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, method)
        else:
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, method)

        maxContour = []
        if len(contours) <= 0:
            return maxContour
        maxArea = 0
        index = 0
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if (area > maxArea):
                maxArea = area
                index = i

        return contours[index]

    def GetOuterContours(mask=None, method=cv2.CHAIN_APPROX_SIMPLE):

        if cv2.__version__ <= '3.4.4':
            img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, method)
        else:
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, method)


        cons_out = []
        for i, cnt in enumerate(contours):
            if hierarchy[0][i][3] == -1:
                cons_out.append(cnt)

        return cons_out

    def GetInformationOfOuterContours(mask):
        #print(cv2.__version__)
        if cv2.__version__ <= '3.4.4':
            img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        numberOut = 0
        areaOut = 0
        for i, cnt in enumerate(contours):
            if len(cnt) < 5:
                continue

            if hierarchy[0][i][3] == -1:
                numberOut += 1
                areaOut += cv2.contourArea(cnt)

        return numberOut, areaOut

    def extractPhenotypes(image, resMask):
        '''
        输入image和对应mask，然后计算通用指标：投影面积/颜色值H
        :param image:
        :param resMask:
        :return:
        singleobjects: 依次列出所有独立轮廓，最后一个所有轮廓的凸包(现在少了一个，凸包暂时不予考虑）
         areas:面积值列表
         hsv_mean_list：HSV颜色值列表
         rgb_mean_list：RGB颜色值列表
        '''
        imagergb = image.copy()
        imagehsv = cv2.cvtColor(imagergb, cv2.COLOR_RGB2HSV)
        # imagergb = cv2.cvtColor(imagergb, cv2.COLOR_RGB2HSV)
        mask = resMask  # cv2.imread(maskfile, cv2.IMREAD_GRAYSCALE)

        if cv2.__version__ <= '3.4.4':
            ret, contours, hierarchy = cv2.findContours(resMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours, hierarchy = cv2.findContours(resMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours_inner = []
        contours_outer = []

        # output items
        singleobjects = []
        rgb_mean_list = []
        hsv_mean_list = []
        areas = []
        totalarea = 0
        # for tt in range(0, len(contours)):
        for i, cnt in enumerate(contours):
            if hierarchy[0][i][3] == -1:  # outer
                contours_outer.append(cnt)
                masksub = np.zeros(mask.shape, np.uint8)
                cv2.drawContours(masksub, contours, i, 255, -1)
                meanval, std = cv2.meanStdDev(imagehsv, mask=masksub)
                hsv_mean_list.append(meanval)

                meanval, std = cv2.meanStdDev(imagergb, mask=masksub)
                rgb_mean_list.append(meanval)

                area = cv2.contourArea(cnt)
                areas.append(int(area))
                totalarea += area

                singleobjects.append(contours[i])
            else:
                contours_inner.append(cnt)

        meanval, std = cv2.meanStdDev(imagehsv, mask=mask)
        hsv_mean_list.append(meanval)

        meanval, std = cv2.meanStdDev(imagergb, mask=mask)
        rgb_mean_list.append(meanval)
        areas.append(totalarea)
        return singleobjects, areas, hsv_mean_list, rgb_mean_list

    def calMaskInfo(mask):
        print("calMaskInfo")
        if cv2.__version__ <= '3.4.4':
            ret, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        OuterArea = 0
        OuterPeri = 0
        numberOBJ=0
        InnerArea = 0
        InnerPeri = 0
        numberHoles = 0
        for i, cnt in enumerate(contours):

            if len(cnt) <= 5:
                continue
            if hierarchy[0][i][3] == -1:  # outer
                area = cv2.contourArea(cnt)
                peri = cv2.arcLength(cnt, True)
                OuterArea += area
                OuterPeri += peri
                numberOBJ += 1
            else:  # inner
                area = cv2.contourArea(cnt)
                peri = cv2.arcLength(cnt, True)
                InnerArea += area
                InnerPeri += peri
                numberHoles += 1

        return OuterArea, OuterPeri, numberOBJ, InnerArea, InnerPeri, numberHoles

    def calMyShapePhenotyping(mask, centerp, radius):
        circlemask = np.zeros(mask.shape, np.uint8)
        cv2.circle(circlemask, centerp, radius, 255,1)
        resMask = cv2.bitwise_and(mask, circlemask)

        if cv2.__version__ <= '3.4.4':
            ret, contours, hierarchy = cv2.findContours(resMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours, hierarchy = cv2.findContours(resMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        outContours = []
        for i, cnt in enumerate(contours):
            if hierarchy[0][i][3] == -1:  # outer
                outContours.append(cnt)

        circleBGR = np.zeros((mask.shape[0],mask.shape[1],3), np.uint8)

        len_segments = 0 # 交线长度
        num_segments = 0
        for i, cnt in enumerate(outContours):
            arc = cv2.arcLength(cnt, True)
            len_segments += arc
            num_segments += 1

        len_segments /=2

        srcperi = 2*math.pi*radius
        segmentratio = len_segments / srcperi  # 交线与圆周长比值

        # 注意返回值差异：
        return num_segments, len_segments, segmentratio, outContours

    def calMyAreaPhenotyping(mask, centerp, radius):
        carea=0 #相交区域的面积
        carearatio=0#面积占比
        holenum=0#相交区域空洞数量

        circlemask = np.zeros(mask.shape, np.uint8)
        cv2.circle(circlemask, centerp, radius, 255, -1)
        resMask = cv2.bitwise_and(mask, circlemask)

        if cv2.__version__ <= '3.4.4':
            ret, contours, hierarchy = cv2.findContours(resMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours, hierarchy = cv2.findContours(resMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        outContours = []
        innContours = []
        for i, cnt in enumerate(contours):
            if hierarchy[0][i][3] == -1:  # outer
                outContours.append(cnt)

            else:
                innContours.append(cnt)

        for i, cnt in enumerate(outContours):
            carea += cv2.contourArea(cnt)

        holenum=len(innContours)
        for i, cnt in enumerate(innContours):
            carea -= cv2.contourArea(cnt)

        circleArea = math.pi * radius *radius
        carearatio = carea / circleArea

        # 注意返回值差异：
        return carea, carearatio, holenum


    def plotCurvePTL(data, outfile="", title="", xlabel="", ylabel="", linewidth=2, titlefontsize=24, labelfontsize=24):
        # 直接绘制趋势图！！！
        import matplotlib.pyplot as plt
        if True:
            fig1 = plt.figure(num='fig111111', figsize=(3, 3), dpi=90, facecolor='#FFFFFF', edgecolor='#0000FF')
            plt.plot(data, linewidth=linewidth, color='black')
            plt.title(title, fontsize=titlefontsize)
            plt.xlabel(xlabel, fontsize=labelfontsize)
            plt.ylabel(ylabel, fontsize=labelfontsize)

            # 去掉刻度
            plt.xticks([])
            plt.yticks([])

            if outfile != "":
                plt.savefig(outfile)
            #plt.show()
            plt.close()

    def plotCurvePTL2(axisXIndice, data, outfile="", title="", xlabel="", ylabel="", linewidth=2, titlefontsize=20, labelfontsize=18):
        # 指定横坐标递增序列
        # 直接绘制趋势图！！！
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import MultipleLocator
        minval = min(data)

        if True:
            fig1 = plt.figure(num='fig', figsize=(3, 3), dpi=200, facecolor='#FFFFFF', edgecolor='#0000FF')

            # figure的百分比,从figure 10%的位置开始绘制, 宽高是figure的80%
            left, bottom, width, height = 0.33, 0.22, 0.62, 0.54
            ax1 = fig1.add_axes([left, bottom, width, height])

            plt.plot(axisXIndice, data, linewidth=linewidth, color='black')
            # 设置图表标题，并给坐标轴加上标签
            plt.title(title, fontsize=titlefontsize)
            plt.xlabel(xlabel, fontsize=labelfontsize)
            plt.ylabel(ylabel, fontsize=labelfontsize)
            # 设置刻度标记的大小
            plt.tick_params(axis='both', which = 'major', labelsize=labelfontsize)
            plt.xlim(axisXIndice[0]-1, axisXIndice[-1]+1)

            days = axisXIndice[-1]-axisXIndice[0]
            tickspace = int(days/2)
            x_major_locator = MultipleLocator(tickspace)# 把x轴的刻度间隔设置为1，并存在变量里
            ax1.xaxis.set_major_locator(x_major_locator)# 把x轴的主刻度设置为1的倍数

            if outfile != "":
                plt.savefig(outfile)
            plt.close()

    def dbsreadtiff(self, imagefile):
        '''
        :param imagefile:
        :return:
        '''
        img = cv2.imread(imagefile, -1)
        print(img.dtype)

        return img


    def get_exif_data(fname):
        """Get embedded EXIF data from image file.
            from PIL import Image
        from PIL.ExifTags import TAGS
        """
        ret = {}
        try:
            img = Image.open(fname)
            if hasattr(img, '_getexif'):
                exifinfo = img._getexif()
                if exifinfo != None:
                    for tag, value in exifinfo.items():
                        decoded = TAGS.get(tag, tag)
                        ret[decoded] = value
        except IOError:
            print
            'IOERROR ' + fname

        print("image prop:", ret)
        return ret

