#!/usr/bin/env python
# encoding: utf-8
'''
@author: du jianjun
@license: (C) Copyright 2019, PERSUPER, NERCITA.
@contact: dujianjun18@126.com
@software: wspp
@file: _micro_phenotypes_table.py
@time: 2020-01-31 3:39
@desc:
all micro phenopyting indexes：

'''

import _pickle as cPickle
from dbsFunction import *


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#from _micro_vbs_parameters import *

def GetTotalArea(cons):
    totalarea = 0
    for i, cnt in enumerate(cons):
        totalarea += cv2.contourArea(cnt)
    return totalarea

def GetOuterContours(mask=None):
    if cv2.__version__ <= '3.4.4':
        img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cons_out = []
    for i, cnt in enumerate(contours):
        if hierarchy[0][i][3] == -1:
            # 无父级别
            cons_out.append(cnt)

    return cons_out

def GetConvexHullFromMask(mask):
    # discarded.....
    # contours = micro_pheno_tables.GetOuterContours(mask)

    if cv2.__version__ <= '3.4.4':
        ret, cons, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    else:
        cons, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # print("contours:", len(contours), "==", contours)
    #print("cons:", len(cons), "==", cons)
    ptsall = []
    for i in range(len(cons)):
        #print("con:", i, "len:", len(cons[i]), "==", cons[i])
        # hull = cv2.convexHull(cnt)
        for j in range(len(cons[i])):
            ptsall.append([[cons[i][j][0][0], cons[i][j][0][1]]])
            # ptsall.append([cons[i][j][0][0],cons[i][j][0][1]])

    if len(ptsall) <= 5:
        return ptsall

    #print("ptsall:", len(ptsall), "==", ptsall)
    hull = cv2.convexHull(np.array(ptsall))
    return hull

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
            # 无父级别
            numberOut += 1
            areaOut += cv2.contourArea(cnt)

    return numberOut, areaOut


class micro_pheno_tables:
    def __init__(self, ctfilename, respath):
        # 20200527 取得um单位
        self.pixelsize = 13.5 / 1e3 # 使用mm单位 mm/pixel

        self.respath = respath
        self.imagefilename = ctfilename
        self.caltime=""

        # new added! 20210608
        # 假设文件名字前两位
        self.circum = ""
        self.cultivarName = ""
        self.id_plant = 0
        self.id_internode = 0

        self.plant_density = ""
        self.plant_fertilizer = ""

        # SZ: slice zone
        self.SZ_A=0
        self.SZ_P= 0
        self.SZ_Imean=0
        self.SZ_Istd=0
        self.SZ_LA=0
        self.SZ_SA=0
        self.SZ_CA=0
        self.SZ_CCA=0

        self.SZ_CAR=0  # new 20210526
        self.SZ_LWR = 0


        # EZ: epidermis zone
        self.EZ_A=0
        self.EZ_T = 0

        # 20210608
        self.EZ_T_Fit = 0

        self.EZ_Imean=0
        self.EZ_Istd = 0

        # VB:
        self.VB_N=0
        self.VB_A=0

        self.VB_Aave=0
        self.VB_Pave=0
        self.VB_LAave=0
        self.VB_SAave = 0
        self.VB_CAave = 0
        self.VB_Imean=0
        self.VB_Istd=0
        self.VB_CCAave=0

        self.VB_CAR = 0 # new
        self.VB_LWR = 0 #

        # PZ: periphery zone
        self.PZ_VB_N = 0
        self.PZ_VB_A = 0
        self.PZ_VB_Imean = 0
        self.PZ_VB_Istd = 0
        self.PZ_A=0
        self.PZ_T = 0  # 使用圆拟合
        self.PZ_Imean=0
        self.PZ_Istd = 0


        # IZ: inner zone
        self.IZ_VB_N = 0
        self.IZ_VB_A = 0
        self.IZ_VB_Imean = 0
        self.IZ_VB_Istd = 0
        self.IZ_A=0
        self.IZ_T = 0    # 使用圆拟合
        self.IZ_Imean=0
        self.IZ_Istd = 0

        self.layer_area = []
        self.layer_intensity = []
        self.layer_vbs_number = []
        self.layer_vbs_area = []
        self.layer_vbs_intensity_mean = []
        self.layer_vbs_intensity_std = []


        self.outerCon_epidermis = []
        self.estimate_epidermis_inner = []
        self.inner_contour = []
        self.vbs_contours = []


        # cal vessel in vbs
        self.vbs_vbs_convex_cnts=[] #
        self.vbs_vessel_cnts=[] #
        self.periphery_vbs_vessel_cnts=[]
        self.inner_vbs_vessel_cnts = []

        # VE : vessel
        self.vessel_vbsconvex_area=0 # OTSU二次分割维管束，得到更加紧凑的维管束边缘轮廓的面积，下面的vessel均包含在其中
        self.vessel_total_number=0
        self.vessel_periphery_number=0
        self.vessel_inner_number=0
        self.vessel_total_area=0
        self.vessel_periphery_area=0
        self.vessel_inner_area=0
        self.vessel_vbs_arearatio=0
        self.vessel_vbs_arearatio_periphery=0
        self.vessel_vbs_arearatio_inner = 0

        # new added traits
        self.ARIVB= 0 # 独立区域面积率 Area ratio of individual connected regions.
        self.SRVB = 0 # 非独立区域分裂率 Split ratio of Non-individual regions. #vascular bunldes splitted split_ratio

        # 新添加
        self.PZ_VB_D =0
        self.PZ_VB_CA = 0
        self.PZ_VB_CAR = 0
        self.IZ_VB_D =0
        self.IZ_VB_CA = 0
        self.IZ_VB_CAR = 0

        self.path_phenotyping = ""

        # 保存维管束力学相关指标
        self.dict_vbs_layers = {}
        self.dict_vbs_mechanics = {}


    def pheno_slice(self, slice_contour=[], ctfilename = ""):
        # 计算slice相关表型
        img = dbsFunction.dbsimread(ctfilename, cv2.IMREAD_GRAYSCALE)
        mask = np.zeros(img.shape, np.uint8)
        cv2.drawContours(mask, [slice_contour], -1, 255, -1)

        self.SZ_A=cv2.contourArea(slice_contour)
        self.SZ_P= cv2.arcLength(slice_contour, True)
        mean,std=cv2.meanStdDev(img, mask=mask)
        self.SZ_Imean = mean[0][0]
        self.SZ_Istd = std[0][0]

        rrect = cv2.minAreaRect(slice_contour)

        # Box2D结构rect：（最小外接矩形的中心（x，y），（宽度，高度），旋转角度）
        self.SZ_LA=max(rrect[1][1],rrect[1][0])
        self.SZ_SA=min(rrect[1][1],rrect[1][0])

        hull=cv2.convexHull(slice_contour)
        self.SZ_CA=cv2.contourArea(hull)
        (x,y),radius = cv2.minEnclosingCircle(slice_contour)
        self.SZ_CCA=math.pi * radius *radius

        self.SZ_CAR = self.SZ_A / self.SZ_CA  # new 20210526
        self.SZ_LWR = self.SZ_LA / self.SZ_SA

        return True

    def pheno_periphery_inner(self, ctfilename="", bSaveProcessing = False):
        img = dbsFunction.dbsimread(ctfilename, cv2.IMREAD_GRAYSCALE)
        innermask = np.zeros(img.shape, np.uint8)
        peripherymask = np.zeros(img.shape, np.uint8)

        cv2.drawContours(innermask, [self.inner_contour], -1, 255, -1)
        # 20210604 内部区域改进，是合理的，不影响结果:经验参数
        morph_size = 5
        kernel = np.ones((morph_size, morph_size), np.uint8)
        innermask = cv2.morphologyEx(innermask, cv2.MORPH_ERODE, kernel)

        cv2.drawContours(peripherymask, [self.estimate_epidermis_inner], -1, 255, -1)
        cv2.drawContours(peripherymask, [self.inner_contour], -1, 0, -1)
        vbs_mask = np.zeros(img.shape, np.uint8)
        cv2.drawContours(vbs_mask, self.vbs_contours, -1, 255, -1)
        vbs_innermask = cv2.bitwise_and(innermask, vbs_mask, mask=vbs_mask)
        vbs_peripherymask = cv2.bitwise_and(peripherymask, vbs_mask, mask=vbs_mask)

        if True:            # self.vbs_contours中最小维管束面积
            dMinA = 1e9
            for id,con in enumerate(self.vbs_contours):
                a = cv2.contourArea(con)
                if a <dMinA:
                    dMinA = a
            #print("minimum vb area:", dMinA)

            used_innermask = np.zeros(vbs_innermask.shape, np.uint8)
            if cv2.__version__ <= '3.4.4':
                img, contours, hierarchy = cv2.findContours(vbs_innermask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            else:
                contours, hierarchy = cv2.findContours(vbs_innermask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for id, con in enumerate(contours):
                a = cv2.contourArea(con)
                if a > dMinA:
                    cv2.drawContours(used_innermask, [con], -1, 255, -1)
            vbs_innermask = used_innermask.copy() # 修改
        ctfilename = ctfilename.replace('\\', '/')
        self.respath = self.respath.replace('\\', '/')

        self.IZ_VB_N = 0
        self.IZ_VB_A = 0
        self.IZ_VB_Imean = 0
        self.IZ_VB_Istd = 0
        self.IZ_A = 0
        self.IZ_T = 0
        self.IZ_Imean= 0
        self.IZ_Istd = 0
        # new
        self.IZ_VB_D =0
        self.IZ_VB_CA = 0
        self.IZ_VB_CAR = 0

        if True:
            if cv2.__version__ <= '3.4.4':
                img, contours, hierarchy = cv2.findContours(vbs_innermask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            else:
                contours, hierarchy = cv2.findContours(vbs_innermask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for i, cnt in enumerate(contours):
                if len(cnt) < 5:
                    continue

                if hierarchy[0][i][3] == -1:
                    # 无父级别
                    self.IZ_VB_N += 1
                    self.IZ_VB_A += cv2.contourArea(cnt)

                    hull = cv2.convexHull(cnt)
                    self.IZ_VB_CA += cv2.contourArea(hull)
                    self.IZ_VB_CAR += cv2.contourArea(cnt) / cv2.contourArea(hull)

        # NOTE:
        self.IZ_A = cv2.contourArea(self.inner_contour)
        self.IZ_VB_D = self.IZ_VB_N/self.IZ_A
        if self.IZ_VB_N>0:
            self.IZ_VB_CAR /= self.IZ_VB_N

        mean, std = cv2.meanStdDev(img, mask=vbs_innermask)
        self.IZ_VB_Imean = mean[0][0]
        self.IZ_VB_Istd = std[0][0]
        self.IZ_T = math.sqrt(self.IZ_A / math.pi)
        mean, std = cv2.meanStdDev(img, mask=innermask)
        self.IZ_Imean= mean[0][0]
        self.IZ_Istd = std[0][0]

        self.PZ_VB_N = 0
        self.PZ_VB_A = 0
        self.PZ_VB_Imean = 0
        self.PZ_VB_Istd = 0
        self.PZ_A = 0
        self.PZ_T = 0
        self.PZ_Imean = 0
        self.PZ_Istd = 0

        self.PZ_VB_D =0
        self.PZ_VB_CA = 0
        self.PZ_VB_CAR = 0

        if True:
            if cv2.__version__ <= '3.4.4':
                img, contours, hierarchy = cv2.findContours(vbs_peripherymask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            else:
                contours, hierarchy = cv2.findContours(vbs_peripherymask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for i, cnt in enumerate(contours):
                if len(cnt) < 5:
                    continue

                if hierarchy[0][i][3] == -1:

                    # 无父级别
                    self.PZ_VB_N += 1
                    self.PZ_VB_A += cv2.contourArea(cnt)

                    hull = cv2.convexHull(cnt)
                    self.PZ_VB_CA += cv2.contourArea(hull)
                    self.PZ_VB_CAR += cv2.contourArea(cnt) / cv2.contourArea(hull)
        self.PZ_A = cv2.contourArea(self.estimate_epidermis_inner) - cv2.contourArea(self.inner_contour)
        self.PZ_VB_D = self.PZ_VB_N/self.PZ_A
        if self.PZ_VB_N>0:
            self.PZ_VB_CAR /= self.PZ_VB_N

        mean, std = cv2.meanStdDev(img, mask=vbs_peripherymask)
        self.PZ_VB_Imean = mean[0][0]
        self.PZ_VB_Istd = std[0][0]

        self.PZ_T = math.sqrt(
            cv2.contourArea(self.estimate_epidermis_inner) / math.pi) - self.IZ_T
        mean, std = cv2.meanStdDev(img, mask=peripherymask)

        self.PZ_Imean = mean[0][0]
        self.PZ_Istd = std[0][0]
        if bSaveProcessing:
            (root, filename) = os.path.split(ctfilename)
            (onlyname, extname) = os.path.splitext(filename)
            resfile = os.path.join(self.respath, onlyname+"_innermask.png")
            dbsFunction.dbsimwrite(resfile, innermask)
            resfile = os.path.join(self.respath, onlyname+"_peripherymask.png")
            dbsFunction.dbsimwrite(resfile, peripherymask)
            resfile = os.path.join(self.respath, onlyname+"_vbs_innermask.png")
            dbsFunction.dbsimwrite(resfile, vbs_innermask)
            resfile = os.path.join(self.respath, onlyname+"_vbs_peripherymask.png")
            dbsFunction.dbsimwrite(resfile, vbs_peripherymask)
        return

    def pheno_epidermis(self, epidermis_contour_out=[],epidermis_contour_in=[], ctfilename = ""):
        img = dbsFunction.dbsimread(ctfilename, cv2.IMREAD_GRAYSCALE)
        mask = np.zeros(img.shape, np.uint8)
        cv2.drawContours(mask, [epidermis_contour_out], -1, 255, -1)
        cv2.drawContours(mask, [epidermis_contour_in], -1, 0, -1)

        # 估计表皮厚度问题
        self.EZ_A=cv2.contourArea(epidermis_contour_out) - cv2.contourArea(epidermis_contour_in)
        mean, std = cv2.meanStdDev(img, mask=mask)
        self.EZ_Imean = mean[0][0]
        self.EZ_Istd = std[0][0]

        len1 = cv2.arcLength(epidermis_contour_out, True)
        len2 = cv2.arcLength(epidermis_contour_in, True)
        len = (len1+len2)/2

        self.EZ_T = self.EZ_A / len
        #rint("epidermis area:{} len:{} EZ_T: {}, EZ_T_Fit:{}".format(self.EZ_A,
        #                                                                                          len,
        #                                                                                          self.EZ_T,
        #                                                                                          self.EZ_T_Fit
        #                                                                                          ))


        return True


    def pheno_vbs(self, vbs_contours=[], ctfilename="", bSaveProcessing = False):
        if len(vbs_contours) <=0:
            return

        img = dbsFunction.dbsimread(ctfilename, cv2.IMREAD_GRAYSCALE)
        mask = np.zeros(img.shape, np.uint8)
        cv2.drawContours(mask, vbs_contours, -1, 255, -1)

        self.VB_N=len(vbs_contours)
        self.VB_A=0
        self.VB_Pave=0
        self.VB_LAave=0
        self.VB_SAave = 0
        self.VB_CAave = 0
        self.VB_CCAave=0
        self.VB_CAR = 0 # new
        self.VB_LWR = 0 #
        for i,cnt in enumerate(vbs_contours):

            area = cv2.contourArea(cnt)
            self.VB_A += area

            peri = cv2.arcLength(cnt, True)
            self.VB_Pave += peri

            rrect = cv2.minAreaRect(cnt)
            wid1 =  max(rrect[1][1], rrect[1][0])
            wid2 = min(rrect[1][1], rrect[1][0])

            self.VB_LAave += wid1
            self.VB_SAave += wid2

            hull = cv2.convexHull(cnt)
            caarea = cv2.contourArea(hull)
            self.VB_CAave += caarea

            (x, y), radius = cv2.minEnclosingCircle(cnt)
            cca = math.pi * radius * radius
            self.VB_CCAave += cca

            self.VB_CAR += area / caarea   # new
            self.VB_LWR = wid1 / wid2#


        self.VB_Aave=self.VB_A / self.VB_N
        self.VB_Pave = self.VB_Pave / self.VB_N

        self.VB_LAave=self.VB_LAave / self.VB_N
        self.VB_SAave = self.VB_SAave / self.VB_N
        self.VB_CAave = self.VB_CAave / self.VB_N
        self.VB_CCAave=self.VB_CCAave / self.VB_N

        self.VB_CAR=self.VB_CAR / self.VB_N
        self.VB_LWR=self.VB_LWR / self.VB_N

        # cal vessel in vbs
        self.vbs_vbs_convex_cnts=[] #
        self.vbs_vessel_cnts=[] #
        self.periphery_vbs_vessel_cnts=[]
        self.inner_vbs_vessel_cnts = []

        imgrgb = dbsFunction.dbsimread(ctfilename, cv2.IMREAD_COLOR)
        for i, cnt in enumerate(vbs_contours):
            #print("NO.", i)
            if len(cnt) <=5:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            vbmask = np.zeros((h,w), np.uint8)
            cv2.drawContours(vbmask, [cnt], -1, 255, -1, offset=(-x,-y))
            ret, dst = cv2.threshold(img[y:y+h,x:x+w], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Note:
            morph_size = 3
            kernel = np.ones((morph_size, morph_size), np.uint8)
            #dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)

            dst = cv2.erode(dst, (morph_size, morph_size))
            dst = cv2.dilate(dst, (morph_size, morph_size))

            hullcon=GetConvexHullFromMask(mask=dst)
            if len(hullcon) <=5:
                continue

            x2, y2, w2, h2 = cv2.boundingRect(cnt)
            ret, dst = cv2.threshold(img[y2:y2+h2,x2:x2+w2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            vbhullmask = np.zeros((h,w), np.uint8)
            cv2.drawContours(vbhullmask,[hullcon],-1, 255,-1)
            realholemask = cv2.bitwise_not(dst,mask=vbhullmask)
            morph_size = 3
            kernel = np.ones((morph_size, morph_size), np.uint8)

            realholemask = cv2.erode(realholemask, (morph_size, morph_size))
            realholemask = cv2.dilate(realholemask, (morph_size, morph_size))

            cntholes = GetOuterContours(realholemask)
            if len(cnt) > 0:cv2.drawContours(imgrgb,[cnt], -1, (255,255,255), 1)
            if len(hullcon) > 0:cv2.drawContours(imgrgb,[hullcon], -1, (255,0,0), 1, offset=(x,y))
            #if len(hull) > 0:cv2.drawContours(imgrgb,[hull], -1, (0,255,0), 1, offset=(x,y))
            if len(cntholes) > 0: cv2.drawContours(imgrgb,cntholes, -1, (0,0,255), 1, offset=(x,y))

            self.vbs_vbs_convex_cnts.append(hullcon)
            for vessel in cntholes:
                vessel += (x,y)
                self.vbs_vessel_cnts.append(vessel)

                dist = cv2.pointPolygonTest(self.inner_contour, (x+w/2, y+h/2), True)
                if dist > 0: # inner
                    self.inner_vbs_vessel_cnts.append(vessel)
                else:
                    self.periphery_vbs_vessel_cnts.append(vessel)

        if bSaveProcessing:
            cv2.drawContours(imgrgb, self.vbs_vessel_cnts, -1, (0,255,255), 1)
            cv2.drawContours(imgrgb, self.inner_vbs_vessel_cnts, -1, (255,0,255), -1)
            cv2.drawContours(imgrgb, self.periphery_vbs_vessel_cnts, -1, (255,255,0), -1)

            (root, filename) = os.path.split(ctfilename)
            #print("root:{} filename:{}".format(root, filename))
            (onlyname, extname) = os.path.splitext(filename)
            #print("onlyname:{} extname:{}".format(onlyname, extname))

            resfile = os.path.join(self.respath, onlyname+"_vb_hole.png")
            dbsFunction.dbsimwrite(resfile, imgrgb)


        mean,std=cv2.meanStdDev(img, mask=mask)
        self.VB_Imean = mean[0][0]
        self.VB_Istd = std[0][0]
        self.vessel_vbsconvex_area = 0
        self.vessel_total_number=0
        self.vessel_periphery_number=0
        self.vessel_inner_number=0
        self.vessel_total_area=0
        self.vessel_periphery_area=0
        self.vessel_inner_area=0
        self.vessel_vbs_arearatio=0
        self.vessel_vbs_arearatio_periphery=0
        self.vessel_vbs_arearatio_inner = 0
        self.vessel_vbsconvex_area = GetTotalArea(self.vbs_vbs_convex_cnts)
        self.vessel_total_number=len(self.vbs_vessel_cnts)
        self.vessel_periphery_number=len(self.periphery_vbs_vessel_cnts)
        self.vessel_inner_number=len(self.inner_vbs_vessel_cnts)
        self.vessel_total_area=GetTotalArea(self.vbs_vessel_cnts)
        self.vessel_periphery_area=GetTotalArea(self.periphery_vbs_vessel_cnts)
        self.vessel_inner_area=GetTotalArea(self.inner_vbs_vessel_cnts)

        if self.VB_A>0: self.vessel_vbs_arearatio=self.vessel_total_area / float(self.VB_A)

        self.vessel_vbs_arearatio_periphery=0
        self.vessel_vbs_arearatio_inner = 0
        if self.PZ_VB_A >0:
            self.vessel_vbs_arearatio_periphery=self.vessel_periphery_area/float(self.PZ_VB_A)
        if self.IZ_VB_A>0:
            self.vessel_vbs_arearatio_inner = self.vessel_inner_area/float(self.IZ_VB_A)

        pass

    def distance(self, pt1=[], pt2=[]):
        return math.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))

    def distance_pt2line(self, point, linept1, linept2):
        point = np.array(point)
        linept1 = np.array(linept1)
        linept2 = np.array(linept2)

        vec1 = linept1 - point
        vec2 = linept2 - point

        dis = np.abs( np.cross(vec1, vec2)) / np.linalg.norm(linept1 - linept2)
        return dis

    def pheno_layers(self,
                     vbs_contours=[],
                     epidermis_contour_out=[],
                     epidermis_contour_in=[],
                     ctfilename=""):
        # 20200908 重新考虑分层计算，得到力学相关性状
        # 进行分层表示（按照
        M= cv2.moments(epidermis_contour_out)
        cX = int(M["m10"]/ M["m00"])
        cY = int(M["m01"] / M["m00"])

        visimage = dbsFunction.dbsimread(ctfilename, cv2.IMREAD_COLOR)
        cv2.drawContours(visimage, [epidermis_contour_out], -1, (255,0,0), 5)
        cv2.drawContours(visimage, [epidermis_contour_in], -1, (255, 255, 0), 5)
        cv2.drawMarker(visimage, (cX, cY), (0,0,255), cv2.MARKER_DIAMOND, markerSize=30, thickness=5)
        center, radius = cv2.minEnclosingCircle(np.concatenate(epidermis_contour_out, 0))
        centerp = (int(center[0]), int(center[1]))
        cv2.drawMarker(visimage, centerp, (0,0,255), cv2.MARKER_CROSS, markerSize=30, thickness=5)
        cv2.circle(visimage, centerp, int(radius), (255,255,255), 3)
        rrect = cv2.minAreaRect(epidermis_contour_out)
        box = cv2.boxPoints(rrect)
        box = np.int0(box)
        (obbx, obby) = rrect[0]
        (obbw, obbh) = rrect[1]
        obbangle = rrect[2]
        len_short = min(rrect[1][0], rrect[1][1])
        dis_01 = self.distance(box[0], box[1])

        pt_l1 = None
        pt_l2 = None
        pt_s1 = None
        pt_s2 = None

        if abs(dis_01 - len_short) < 1:
            pt_s1 = (box[0] + box[1]) / 2
            pt_s2 = (box[3] + box[2]) / 2
            pt_s1 = (int(pt_s1[0]), int(pt_s1[1]))
            pt_s2 = (int(pt_s2[0]), int(pt_s2[1]))
            cv2.line(visimage, (pt_s1[0], pt_s1[1]), (pt_s2[0], pt_s2[1]), (255, 0, 0), 4)

            # 长边，中点连线形成短轴
            pt_l1 = (box[0] + box[3])/2
            pt_l2 = (box[1] + box[2])/2
            pt_l1 = (int(pt_l1[0]), int(pt_l1[1]))
            pt_l2 = (int(pt_l2[0]), int(pt_l2[1]))
            cv2.line(visimage, (pt_l1[0],pt_l1[1]), (pt_l2[0],pt_l2[1]), (0,0,255), 4)
        else:
            pt_l1 = (box[0] + box[1]) / 2
            pt_l2 = (box[3] + box[2]) / 2
            pt_l1 = (int(pt_l1[0]), int(pt_l1[1]))
            pt_l2 = (int(pt_l2[0]), int(pt_l2[1]))

            cv2.line(visimage, (pt_l1[0], pt_l1[1]), (pt_l2[0], pt_l2[1]), (255, 0, 0), 4)

            pt_s1 = (box[0] + box[3])/2
            pt_s2 = (box[1] + box[2])/2
            pt_s1 = (int(pt_s1[0]), int(pt_s1[1]))
            pt_s2 = (int(pt_s2[0]), int(pt_s2[1]))
            cv2.line(visimage, (pt_s1[0],pt_s1[1]), (pt_s2[0],pt_s2[1]), (0,0,255), 4)

        cv2.polylines(visimage, [box], True, (0,255,0), 4)
        nSegment = 5
        dis_Segment = radius / nSegment
        total_layer_vbs = []
        total_layer_vbs_area = 0
        total_layer_PolarMomentOfInertia = 0
        total_layer_MomentOfArea = 0
        total_layer_MomentOfInertiaOfArea_L = 0
        total_layer_MomentOfInertiaOfArea_S = 0

        for i in range(1, nSegment + 1):
            radius_ = i * dis_Segment
            cv2.circle(visimage, centerp, int(radius_), (255, 0, 255), 3)

            # 转动惯量
            # Moment of inertia.
            layer_vbs = []
            layer_vbs_area = 0
            layer_PolarMomentOfInertia = 0
            layer_MomentOfArea = 0
            layer_MomentOfInertiaOfArea_L = 0 # 大的惯性矩
            layer_MomentOfInertiaOfArea_S = 0

            for con in vbs_contours:
                vbM = cv2.moments(con)
                vbX = int(vbM["m10"] / vbM["m00"])
                vbY = int(vbM["m01"] / vbM["m00"])

                dis = self.distance( (vbX, vbY), (cX, cY))
                radius_in = (i-1) * dis_Segment
                if dis < radius_ and dis >= radius_in:
                    layer_vbs.append(con)
                    cv_area = cv2.contourArea(con)
                    layer_vbs_area += cv_area

                    layer_PolarMomentOfInertia += cv_area * dis * dis
                    layer_MomentOfArea += cv_area * dis


                    dis_L = self.distance_pt2line((vbX, vbY), pt_l1, pt_l2)
                    dis_S = self.distance_pt2line((vbX, vbY), pt_s1, pt_s2)
                    MOIA_L = cv_area * dis_L * dis_L
                    MOIA_S = cv_area * dis_S * dis_S

                    layer_MomentOfInertiaOfArea_L += MOIA_L  # 大的惯性矩
                    layer_MomentOfInertiaOfArea_S += MOIA_S

                    total_layer_vbs.append(con)
                    total_layer_vbs_area += cv_area
                    total_layer_PolarMomentOfInertia += cv_area * dis * dis
                    total_layer_MomentOfArea += cv_area * dis
                    total_layer_MomentOfInertiaOfArea_L +=MOIA_L  # 大的惯性矩
                    total_layer_MomentOfInertiaOfArea_S +=MOIA_S

            num_vbs = len(layer_vbs)
            avearea_vbs = 0
            avePMOI = 0
            aveMOA = 0
            aveMOIAL=0
            aveMOIAS = 0
            if num_vbs > 0:
                avearea_vbs = layer_vbs_area/num_vbs
                avePMOI = layer_PolarMomentOfInertia/num_vbs
                aveMOA = layer_MomentOfArea / num_vbs

                aveMOIAL = layer_MomentOfInertiaOfArea_L / num_vbs
                aveMOIAS = layer_MomentOfInertiaOfArea_S / num_vbs

                txt = "Layer%d NUM:%d PMOI:%.2fmm4 MOA:%.2fmm3 MOIAL:%.2fmm4 MOIA_S:%.2fmm4"%(i, num_vbs,
                                                                                  layer_PolarMomentOfInertia*self.pixelsize*self.pixelsize*self.pixelsize*self.pixelsize,
                                                                                  layer_MomentOfArea*self.pixelsize*self.pixelsize*self.pixelsize,
                                                                                  layer_MomentOfInertiaOfArea_L*self.pixelsize*self.pixelsize*self.pixelsize*self.pixelsize,
                                                                                  layer_MomentOfInertiaOfArea_S*self.pixelsize*self.pixelsize*self.pixelsize*self.pixelsize
                                                                                  )
                cv2.putText(visimage, txt, (10, i*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)


            self.dict_vbs_layers[i] = {
                "layer_vbs":layer_vbs,
                "layer_vbs_num": num_vbs,
                "layer_vbs_area": layer_vbs_area,
                "layer_vbs_avearea": avearea_vbs,
                "layer_vbs_PMOI": layer_PolarMomentOfInertia,
                "layer_vbs_avePMOI": avePMOI,
                "layer_vbs_MOA": layer_MomentOfArea,
                "layer_vbs_aveMOA": aveMOA,

                "layer_vbs_MOIAL": layer_MomentOfInertiaOfArea_L,
                "layer_vbs_aveMOIAL": aveMOIAL,
                "layer_vbs_MOIAS": layer_MomentOfInertiaOfArea_S,
                "layer_vbs_aveMOIAS": aveMOIAS,
            }

        # 所有层
        num_vbs = len(total_layer_vbs)
        avearea_vbs = 0
        avePMOI = 0
        aveMOA = 0
        aveMOIAL=0
        aveMOIAS = 0
        if num_vbs > 0:
            avearea_vbs = total_layer_vbs_area/num_vbs
            avePMOI = total_layer_PolarMomentOfInertia/num_vbs
            aveMOA = total_layer_MomentOfArea / num_vbs

            aveMOIAL = total_layer_MomentOfInertiaOfArea_L / num_vbs
            aveMOIAS = total_layer_MomentOfInertiaOfArea_S / num_vbs

            txt = "Total VBS NUM:%d PMOI:%.2fmm4 MOA:%.2fmm3 MOIAL:%.2fmm4 MOIAS:%.2fmm4"%(num_vbs,
                                                                              total_layer_PolarMomentOfInertia*self.pixelsize*self.pixelsize*self.pixelsize*self.pixelsize,
                                                                              total_layer_MomentOfArea*self.pixelsize*self.pixelsize*self.pixelsize,
                                                                              total_layer_MomentOfInertiaOfArea_L*self.pixelsize*self.pixelsize*self.pixelsize*self.pixelsize,
                                                                              total_layer_MomentOfInertiaOfArea_S*self.pixelsize*self.pixelsize*self.pixelsize*self.pixelsize
                                                                              )
            cv2.putText(visimage, txt, (10, 2000-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.putText(visimage, "PMOI:PolarMomentOfInertia", (10, 2000-80), cv2.FONT_HERSHEY_SIMPLEX, 1, (127,127,127), 2)
            cv2.putText(visimage, "MOA: MomentOfArea", (10, 2000-120), cv2.FONT_HERSHEY_SIMPLEX, 1, (127,127,127), 2)
            cv2.putText(visimage, "MOIA:MomentOfInertiaOfArea or SecondMomentOfArea", (10, 2000-160), cv2.FONT_HERSHEY_SIMPLEX,1, (127,127,127), 2)
            cv2.putText(visimage, "MOIAL(Blue)", (10, 2000-200), cv2.FONT_HERSHEY_SIMPLEX, 1, (127,127,127), 2)
            cv2.putText(visimage, "MOIAS(Red)", (10, 2000 - 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (127,127,127), 2)

            self.dict_vbs_mechanics = {
                "VBS_N":num_vbs,
                "avearea_vbs":avearea_vbs,
                "avePMOI":avePMOI,
                "aveMOA":aveMOA,
                "aveMOIAL":aveMOIAL,
                "aveMOIAS":aveMOIAS,
                "PMOI":total_layer_PolarMomentOfInertia,
                "MOA":total_layer_MomentOfArea,
                "MOIAL":total_layer_MomentOfInertiaOfArea_L,
                "MOIAS":total_layer_MomentOfInertiaOfArea_S
            }

        
        (root, filename) = os.path.split(ctfilename)
        (onlyname, extname) = os.path.splitext(filename)
        resfile = os.path.join(self.respath, onlyname + "_layers.jpg")

        dbsFunction.dbsimwrite(resfile, visimage, ".jpg")
        pass

    def getValues(self):
        out_values = []
        out_values.append(self.circum)
        out_values.append(self.id_plant)

        out_values.append(self.cultivarName)
        out_values.append(self.id_internode)

        out_values.append(self.plant_density)
        out_values.append(self.plant_fertilizer)
        out_values.append(self.imagefilename)
        out_values.append(self.caltime)

        out_values.append(self.SZ_A)
        out_values.append(self.SZ_P)
        out_values.append(self.SZ_Imean)
        out_values.append(self.SZ_Istd)
        out_values.append(self.SZ_LA)
        out_values.append(self.SZ_SA)
        out_values.append(self.SZ_CA)
        out_values.append(self.SZ_CCA)

        out_values.append(self.SZ_CAR)
        out_values.append(self.SZ_LWR)


        out_values.append(self.EZ_A)
        out_values.append(self.EZ_T)

        out_values.append(self.EZ_Imean)
        out_values.append(self.EZ_Istd)

        #self.PZ_A)
        #self.PZ_T = 0
        #self.PZ_Imean=0
        #self.PZ_Istd = 0

        #self.VB_N=0
        #self.VB_A=0

        out_values.append(self.VB_Aave)
        out_values.append(self.VB_Pave)
        out_values.append(self.VB_LAave)
        out_values.append(self.VB_SAave)
        out_values.append(self.VB_CAave)
        out_values.append(self.VB_Imean)
        out_values.append(self.VB_Istd)
        out_values.append(self.VB_CCAave)

        out_values.append(self.VB_CAR)
        out_values.append(self.VB_LWR)


        #self.layer_area = []
        #self.layer_intensity = []
        #self.layer_vbs_number = []
        #self.layer_vbs_area = []
        #self.layer_vbs_intensity_mean = []
        #self.layer_vbs_intensity_std = []

        #prename,extname = os.path.splitext(ctfilename)
        #path_all_boxes_csv = prename + "_calres.cvs"

        return out_values
        with open(csvfilename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # writer.writerow(["index","a_name","b_name"])  # 写入列名，如果没有列名可以不执行这一行
            # writer.writerows([[0, 1, 3], [1, 2, 3], [2, 3, 4]]) # 写入多行用writerows
            #for box in image_local_box:
            writer.writerow(out_values)  # 写入1行用writerow; row_data是你要写入的数据，最好是list类型。
        pass

    def getHeaderAndValue(self):
        out_header = []
        out_values = []
        out_header.append("respath")
        out_values.append(self.respath)


        out_header.append("circum")
        out_values.append(self.circum)
        out_header.append("id_plant")
        out_values.append(self.id_plant)

        out_header.append("cultivarName")
        out_values.append(self.cultivarName)
        out_header.append("id_internode")
        out_values.append(self.id_internode)

        out_header.append("plant_density")
        out_values.append(self.plant_density)
        out_header.append("plant_fertilizer")
        out_values.append(self.plant_fertilizer)



        out_header.append("imagefilename")
        out_values.append(self.imagefilename)
        out_header.append("caltime")
        out_values.append(self.caltime)

        out_header.append("SZ_A")
        out_values.append(self.SZ_A)
        out_header.append("SZ_P")
        out_values.append(self.SZ_P)
        out_header.append("SZ_Imean")
        out_values.append(self.SZ_Imean)
        out_header.append("SZ_Istd")
        out_values.append(self.SZ_Istd)
        out_header.append("SZ_LA")
        out_values.append(self.SZ_LA)
        out_header.append("SZ_SA")
        out_values.append(self.SZ_SA)
        out_header.append("SZ_CA")
        out_values.append(self.SZ_CA)
        out_header.append("SZ_CCA")
        out_values.append(self.SZ_CCA)


        out_header.append("SZ_CAR")
        out_values.append(self.SZ_CAR)
        out_header.append("SZ_LWR")
        out_values.append(self.SZ_LWR)



        out_header.append("EZ_A")
        out_values.append(self.EZ_A)

        out_header.append("EZ_T")
        out_values.append(self.EZ_T)

        out_header.append("EZ_T_Fit")
        out_values.append(self.EZ_T_Fit)

        out_header.append("EZ_Imean")
        out_values.append(self.EZ_Imean)
        out_header.append("EZ_Istd")
        out_values.append(self.EZ_Istd)

        out_header.append("VB_N")
        out_values.append(self.VB_N)
        out_header.append("VB_A")
        out_values.append(self.VB_A)

        out_header.append("VB_Aave")
        out_values.append(self.VB_Aave)
        out_header.append("VB_Pave")
        out_values.append(self.VB_Pave)
        out_header.append("VB_LAave")
        out_values.append(self.VB_LAave)
        out_header.append("VB_SAave")
        out_values.append(self.VB_SAave)
        out_header.append("VB_CAave")
        out_values.append(self.VB_CAave)
        out_header.append("VB_Imean")
        out_values.append(self.VB_Imean)
        out_header.append("VB_Istd")
        out_values.append(self.VB_Istd)
        out_header.append("VB_CCAave")
        out_values.append(self.VB_CCAave)

        out_header.append("VB_CAR")
        out_values.append(self.VB_CAR)
        out_header.append("VB_LWR")
        out_values.append(self.VB_LWR)


        out_header.append("PZ_VB_N")
        out_values.append(self.PZ_VB_N)
        out_header.append("PZ_VB_A")
        out_values.append(self.PZ_VB_A)
        out_header.append("PZ_VB_Imean")
        out_values.append(self.PZ_VB_Imean)
        out_header.append("PZ_VB_Istd")
        out_values.append(self.PZ_VB_Istd)
        out_header.append("PZ_A")
        out_values.append(self.PZ_A)
        out_header.append("PZ_T")  # 使用圆拟合
        out_values.append(self.PZ_T)  # 使用圆拟合
        out_header.append("PZ_Imean")
        out_values.append(self.PZ_Imean)
        out_header.append("PZ_Istd")
        out_values.append(self.PZ_Istd)

        out_header.append("IZ_VB_N")
        out_values.append(self.IZ_VB_N)
        out_header.append("IZ_VB_A")
        out_values.append(self.IZ_VB_A)
        out_header.append("IZ_VB_Imean")
        out_values.append(self.IZ_VB_Imean)
        out_header.append("IZ_VB_Istd")
        out_values.append(self.IZ_VB_Istd)
        out_header.append("IZ_A")
        out_values.append(self.IZ_A)
        out_header.append("IZ_T")    # 使用圆拟合
        out_values.append(self.IZ_T)    # 使用圆拟合
        out_header.append("IZ_Imean")
        out_values.append(self.IZ_Imean)
        out_header.append("IZ_Istd")
        out_values.append(self.IZ_Istd)

        out_header.append("vessel_vbsconvex_area") # OTSU二次分割维管束，得到更加紧凑的维管束边缘轮廓的面积，下面的vessel均包含在其中
        out_values.append(self.vessel_vbsconvex_area) # OTSU二次分割维管束，得到更加紧凑的维管束边缘轮廓的面积，下面的vessel均包含在其中
        out_header.append("vessel_total_number")
        out_values.append(self.vessel_total_number)
        out_header.append("vessel_periphery_number")
        out_values.append(self.vessel_periphery_number)
        out_header.append("vessel_inner_number")
        out_values.append(self.vessel_inner_number)
        out_header.append("vessel_total_area")
        out_values.append(self.vessel_total_area)
        out_header.append("vessel_periphery_area")
        out_values.append(self.vessel_periphery_area)
        out_header.append("vessel_inner_area")
        out_values.append(self.vessel_inner_area)
        out_header.append("vessel_vbs_arearatio")
        out_values.append(self.vessel_vbs_arearatio)
        out_header.append("vessel_vbs_arearatio_periphery")
        out_values.append(self.vessel_vbs_arearatio_periphery)
        out_header.append("vessel_vbs_arearatio_inner")
        out_values.append(self.vessel_vbs_arearatio_inner)


        out_header.append("ARIVB")
        out_values.append(self.ARIVB)
        out_header.append("SRVB")
        out_values.append(self.SRVB)


        out_header.append("PZ_VB_D")
        out_values.append(self.PZ_VB_D)
        out_header.append("PZ_VB_CA")
        out_values.append(self.PZ_VB_CA)
        out_header.append("PZ_VB_CAR")
        out_values.append(self.PZ_VB_CAR)
        out_header.append("IZ_VB_D")
        out_values.append(self.IZ_VB_D)
        out_header.append("IZ_VB_CA")
        out_values.append(self.IZ_VB_CA)
        out_header.append("IZ_VB_CAR")
        out_values.append(self.IZ_VB_CAR)



        return out_header, out_values

    def getHeaderAndValueMM(self):
        # 20200527 取得毫米单位
        out_header = []
        out_values = []
        out_header.append("respath")
        out_values.append(self.respath)


        out_header.append("circum")
        out_values.append(self.circum)
        out_header.append("id_plant")
        out_values.append(self.id_plant)

        out_header.append("cultivarName")
        out_values.append(self.cultivarName)
        out_header.append("id_internode")
        out_values.append(self.id_internode)


        out_header.append("plant_density")
        out_values.append(self.plant_density)
        out_header.append("plant_fertilizer")
        out_values.append(self.plant_fertilizer)

        out_header.append("imagefilename")
        out_values.append(self.imagefilename)
        out_header.append("caltime")
        out_values.append(self.caltime)

        out_header.append("SZ_A")
        out_values.append(self.SZ_A*self.pixelsize*self.pixelsize)
        out_header.append("SZ_P")
        out_values.append(self.SZ_P*self.pixelsize)
        out_header.append("SZ_Imean")
        out_values.append(self.SZ_Imean)
        out_header.append("SZ_Istd")
        out_values.append(self.SZ_Istd)
        out_header.append("SZ_LA")
        out_values.append(self.SZ_LA*self.pixelsize)
        out_header.append("SZ_SA")
        out_values.append(self.SZ_SA*self.pixelsize)
        out_header.append("SZ_CA")
        out_values.append(self.SZ_CA*self.pixelsize*self.pixelsize)
        out_header.append("SZ_CCA")
        out_values.append(self.SZ_CCA*self.pixelsize*self.pixelsize)

        out_header.append("SZ_CAR")
        out_values.append(self.SZ_CAR)
        out_header.append("SZ_LWR")
        out_values.append(self.SZ_LWR)

        out_header.append("EZ_A")
        out_values.append(self.EZ_A*self.pixelsize*self.pixelsize)
        out_header.append("EZ_T")
        out_values.append(self.EZ_T*self.pixelsize)
        out_header.append("EZ_Imean")
        out_values.append(self.EZ_Imean)
        out_header.append("EZ_Istd")
        out_values.append(self.EZ_Istd)

        out_header.append("VB_N")
        out_values.append(self.VB_N)
        out_header.append("VB_A")
        out_values.append(self.VB_A*self.pixelsize*self.pixelsize)

        out_header.append("VB_Aave")
        out_values.append(self.VB_Aave*self.pixelsize*self.pixelsize)
        out_header.append("VB_Pave")
        out_values.append(self.VB_Pave*self.pixelsize)
        out_header.append("VB_LAave")
        out_values.append(self.VB_LAave*self.pixelsize)
        out_header.append("VB_SAave")
        out_values.append(self.VB_SAave*self.pixelsize)
        out_header.append("VB_CAave")
        out_values.append(self.VB_CAave*self.pixelsize*self.pixelsize)
        out_header.append("VB_Imean")
        out_values.append(self.VB_Imean)
        out_header.append("VB_Istd")
        out_values.append(self.VB_Istd)
        out_header.append("VB_CCAave")
        out_values.append(self.VB_CCAave*self.pixelsize*self.pixelsize)

        out_header.append("VB_CAR")
        out_values.append(self.VB_CAR)
        out_header.append("VB_LWR")
        out_values.append(self.VB_LWR)


        out_header.append("PZ_VB_N")
        out_values.append(self.PZ_VB_N)
        out_header.append("PZ_VB_A")
        out_values.append(self.PZ_VB_A*self.pixelsize*self.pixelsize)
        out_header.append("PZ_VB_Imean")
        out_values.append(self.PZ_VB_Imean)
        out_header.append("PZ_VB_Istd")
        out_values.append(self.PZ_VB_Istd)
        out_header.append("PZ_A")
        out_values.append(self.PZ_A*self.pixelsize*self.pixelsize)
        out_header.append("PZ_T")  # 使用圆拟合
        out_values.append(self.PZ_T*self.pixelsize)  # 使用圆拟合
        out_header.append("PZ_Imean")
        out_values.append(self.PZ_Imean)
        out_header.append("PZ_Istd")
        out_values.append(self.PZ_Istd)

        out_header.append("IZ_VB_N")
        out_values.append(self.IZ_VB_N)
        out_header.append("IZ_VB_A")
        out_values.append(self.IZ_VB_A*self.pixelsize*self.pixelsize)
        out_header.append("IZ_VB_Imean")
        out_values.append(self.IZ_VB_Imean)
        out_header.append("IZ_VB_Istd")
        out_values.append(self.IZ_VB_Istd)
        out_header.append("IZ_A")
        out_values.append(self.IZ_A*self.pixelsize*self.pixelsize)
        out_header.append("IZ_T")    # 使用圆拟合
        out_values.append(self.IZ_T*self.pixelsize)    # 使用圆拟合
        out_header.append("IZ_Imean")
        out_values.append(self.IZ_Imean)
        out_header.append("IZ_Istd")
        out_values.append(self.IZ_Istd)

        out_header.append("vessel_vbsconvex_area") # OTSU二次分割维管束，得到更加紧凑的维管束边缘轮廓的面积，下面的vessel均包含在其中
        out_values.append(self.vessel_vbsconvex_area*self.pixelsize*self.pixelsize) # OTSU二次分割维管束，得到更加紧凑的维管束边缘轮廓的面积，下面的vessel均包含在其中
        out_header.append("vessel_total_number")
        out_values.append(self.vessel_total_number)
        out_header.append("vessel_periphery_number")
        out_values.append(self.vessel_periphery_number)
        out_header.append("vessel_inner_number")
        out_values.append(self.vessel_inner_number)
        out_header.append("vessel_total_area")
        out_values.append(self.vessel_total_area*self.pixelsize*self.pixelsize)
        out_header.append("vessel_periphery_area")
        out_values.append(self.vessel_periphery_area*self.pixelsize*self.pixelsize)
        out_header.append("vessel_inner_area")
        out_values.append(self.vessel_inner_area*self.pixelsize*self.pixelsize)
        out_header.append("vessel_vbs_arearatio")
        out_values.append(self.vessel_vbs_arearatio)
        out_header.append("vessel_vbs_arearatio_periphery")
        out_values.append(self.vessel_vbs_arearatio_periphery)

        out_header.append("vessel_vbs_arearatio_inner")
        out_values.append(self.vessel_vbs_arearatio_inner)


        out_header.append("ARIVB")
        out_values.append(self.ARIVB)
        out_header.append("SRVB")
        out_values.append(self.SRVB)

        out_header.append("PZ_VB_D")
        out_values.append(self.PZ_VB_D/self.pixelsize/self.pixelsize)
        out_header.append("PZ_VB_CA")
        out_values.append(self.PZ_VB_CA*self.pixelsize*self.pixelsize)
        out_header.append("PZ_VB_CAR")
        out_values.append(self.PZ_VB_CAR)
        out_header.append("IZ_VB_D")
        out_values.append(self.IZ_VB_D/self.pixelsize/self.pixelsize)
        out_header.append("IZ_VB_CA")
        out_values.append(self.IZ_VB_CA*self.pixelsize*self.pixelsize)
        out_header.append("IZ_VB_CAR")
        out_values.append(self.IZ_VB_CAR)

        return out_header, out_values

    def getHeaderAndValueMM_mechanics(self):
        # 维管束力学指标
        # 20200908 取得毫米单位
        out_header = []
        out_values = []
        out_header.append("respath")
        out_values.append(self.respath)


        out_header.append("circum")
        out_values.append(self.circum)
        out_header.append("id_plant")
        out_values.append(self.id_plant)

        out_header.append("cultivarName")
        out_values.append(self.cultivarName)
        out_header.append("id_internode")
        out_values.append(self.id_internode)


        out_header.append("plant_density")
        out_values.append(self.plant_density)
        out_header.append("plant_fertilizer")
        out_values.append(self.plant_fertilizer)

        out_header.append("imagefilename")
        out_values.append(self.imagefilename)
        out_header.append("caltime")
        out_values.append(self.caltime)
        out_header.append("Layers")
        out_values.append(len(self.dict_vbs_layers))

        for key, valAll in self.dict_vbs_mechanics.items():
            name = key
            val = valAll
            out_header.append(name)
            out_values.append(val)

        for key, valLayer in self.dict_vbs_layers.items():
            name = "VB_N_L{}".format(key)
            val = valLayer["layer_vbs_num"]
            out_header.append(name)
            out_values.append(val)
            
            name = "VB_totalA_L{}".format(key)
            val = valLayer["layer_vbs_area"] *self.pixelsize*self.pixelsize
            out_header.append(name)
            out_values.append(val)
            
            name = "VB_aveA_L{}".format(key)
            val = valLayer["layer_vbs_avearea"]*self.pixelsize*self.pixelsize
            out_header.append(name)
            out_values.append(val)

            name = "VB_PMOI_L{}".format(key)
            val = valLayer["layer_vbs_PMOI"]*self.pixelsize*self.pixelsize*self.pixelsize*self.pixelsize
            out_header.append(name)
            out_values.append(val)

            name = "VB_avePMOI_L{}".format(key)
            val = valLayer["layer_vbs_avePMOI"]*self.pixelsize*self.pixelsize*self.pixelsize*self.pixelsize
            out_header.append(name)
            out_values.append(val)

            name = "VB_MOA_L{}".format(key)
            val = valLayer["layer_vbs_MOA"] *self.pixelsize*self.pixelsize*self.pixelsize
            out_header.append(name)
            out_values.append(val)

            name = "VB_aveMOA_L{}".format(key)
            val = valLayer["layer_vbs_aveMOA"]*self.pixelsize*self.pixelsize*self.pixelsize
            out_header.append(name)
            out_values.append(val)


            name = "VB_MOIAL_L{}".format(key)
            val = valLayer["layer_vbs_MOIAL"] *self.pixelsize*self.pixelsize*self.pixelsize*self.pixelsize
            out_header.append(name)
            out_values.append(val)

            name = "VB_aveMOIAL_L{}".format(key)
            val = valLayer["layer_vbs_aveMOIAL"]*self.pixelsize*self.pixelsize*self.pixelsize*self.pixelsize
            out_header.append(name)
            out_values.append(val)

            name = "VB_MOIAS_L{}".format(key)
            val = valLayer["layer_vbs_MOIAS"]*self.pixelsize*self.pixelsize*self.pixelsize*self.pixelsize
            out_header.append(name)
            out_values.append(val)

            name = "VB_aveMOIAS_L{}".format(key)
            val = valLayer["layer_vbs_aveMOIAS"]*self.pixelsize*self.pixelsize*self.pixelsize*self.pixelsize
            out_header.append(name)
            out_values.append(val)

        return out_header, out_values

    def getHeader():
        out_header = []



        out_header.append("circum")
        out_header.append("id_plant")

        out_header.append("cultivarName")
        out_header.append("id_internode")
        out_header.append("plant_density")
        out_header.append("plant_fertilizer")



        out_header.append('imagefilename')
        out_header.append('caltime')

        out_header.append('SZ_A')
        out_header.append('SZ_P')
        out_header.append('SZ_Imean')
        out_header.append('SZ_Istd')
        out_header.append('SZ_LA')
        out_header.append('SZ_SA')
        out_header.append('SZ_CA')
        out_header.append('SZ_CCA')

        out_header.append('EZ_A')
        out_header.append('EZ_T')
        out_header.append('EZ_T_Fit')
        out_header.append('EZ_Imean')
        out_header.append('EZ_Istd')

        #self.PZ_A)
        #self.PZ_T = 0
        #self.PZ_Imean=0
        #self.PZ_Istd = 0

        #self.VB_N=0
        #self.VB_A=0

        out_header.append('VB_Aave')
        out_header.append('VB_Pave')
        out_header.append('VB_LAave')
        out_header.append('VB_SAave')
        out_header.append('VB_CAave')
        out_header.append('VB_Imean')
        out_header.append('VB_Istd')
        out_header.append('VB_CCAave')

        return out_header
        pass
    def write_serialize_statistic(statistic_cvs_file, micro_pheno_=[]):

        with open(statistic_cvs_file, 'w', newline='') as csvfile:  # 以写入模式打开csv文件，如果没有csv文件会自动创建。
            writer = csv.writer(csvfile)

            for i in range(len(micro_pheno_)):
                headers, values = micro_pheno_[i].getHeaderAndValueMM()
                if i==0:
                    writer.writerow(headers)
                writer.writerow(values)  # 写入1行用writerow; row_data是你要写入的数据，最好是list类型。
        pass
    def write_serialize_statistic_mechanics(statistic_cvs_file_mechanics, micro_pheno_=[]):
        with open(statistic_cvs_file_mechanics, 'w', newline='') as csvfile:  # 以写入模式打开csv文件，如果没有csv文件会自动创建。
            writer = csv.writer(csvfile)

            for i in range(len(micro_pheno_)):

                headers, values = micro_pheno_[i].getHeaderAndValueMM_mechanics()
                if i==0:
                    writer.writerow(headers)
                writer.writerow(values)  # 写入1行用writerow; row_data是你要写入的数据，最好是list类型。
        pass

    def save2pkl(pkl_file, pheno_tables):
        with open( pkl_file, 'wb') as f:
            cPickle.dump(pheno_tables, f)

    def load4pkl(pkl_file):
        if not os.path.exists(pkl_file):
            return None

        with open(pkl_file, 'rb') as f:
            pheno_tables = cPickle.load(f, encoding='bytes')
            print(type(pheno_tables))

            return pheno_tables

    def GetLargestContour(mask, method=cv2.CHAIN_APPROX_SIMPLE):
        if cv2.__version__ <= '3.4.4':
            img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, method)
        else:
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, method)

        # contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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

        maxContour = contours[index]
        return contours[index]

    def GetOuterContours(mask=None, method=cv2.CHAIN_APPROX_SIMPLE):

        if cv2.__version__ <= '3.4.4':
            img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, method)
        else:
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, method)
        cons_out = []
        for i, cnt in enumerate(contours):
            if hierarchy[0][i][3] == -1:
                # 无父级别
                cons_out.append(cnt)

        return cons_out


if __name__ == "__main__":

    exit()
    pkl_file = r"H8\H8.pkl"
    ress = micro_pheno_tables.load4pkl(pkl_file)
    for id, res in enumerate(ress):
        print( "cultivarName:", res.cultivarName )
        print( "id_internode:", res.id_internode )