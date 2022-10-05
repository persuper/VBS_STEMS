#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022-03-28 22:41
# @Author  : DBS
# @Email   : dujianjun18@126.com
# @File    : vbs_api.py
from dbsFunction import *
from _micro_phenotypes_table import micro_pheno_tables

import pandas.io.formats.excel
pandas.io.formats.excel.header_style = None
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from dbsAdaptiveWatershed import dbsAdaptiveWatershed

import matplotlib as mpl
mpl.rcParams['font.family'] = ['Calibri']
mpl.rcParams['axes.unicode_minus']=False
config = 'models/vbs_maizestem/vbs_maizestem_deeplabv3_r50-d8_512x512_20k_voc12aug.py'
checkpoint = 'models/vbs_maizestem/vbs_maizestem_deeplabv3_r50-d8_512x512_20k_voc12aug.pth'
device = 'cuda:0'
model = init_segmentor(config, checkpoint, device=device)

class vbs_maizestem():
    def __init__(self):
        pass

    def DetectEpidermis(self, imagefile, nThickness=7, bshow=False):
        imgGray = dbsFunction.dbsimread(imagefile, cv2.IMREAD_GRAYSCALE)
        bk = np.zeros((imgGray.shape[0], imgGray.shape[1]), np.uint8)
        radius = int(imgGray.shape[0] / 2)
        cv2.circle(bk, (radius, radius), radius, 255, -1)
        meanI, stdI = cv2.meanStdDev(imgGray, mask=bk)
        ret, dst = cv2.threshold(imgGray, int(meanI[0]), 255, cv2.THRESH_BINARY)
        if True:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dst = cv2.dilate(dst, kernel)
            dst = cv2.erode(dst, kernel)

        if cv2.__version__ <= '3.4.4':
            image0, contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) <= 0:
            return None

        maxArea = 0
        indexCon = -1
        for i, cnt in enumerate(contours):
            if len(cnt) < 20:
                continue
            area = cv2.contourArea(cnt)
            if area > maxArea:
                maxArea = area
                indexCon = i

        if indexCon < 0:
            return None

        sliceMask = np.zeros(imgGray.shape, imgGray.dtype)
        cv2.drawContours(sliceMask, contours, indexCon, 255, -1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        sliceMask = cv2.erode(sliceMask, kernel)
        sliceMask = cv2.dilate(sliceMask, kernel)
        sliceMask = cv2.dilate(sliceMask, kernel)
        sliceMask = cv2.dilate(sliceMask, kernel)
        sliceMask = cv2.erode(sliceMask, kernel)
        sliceMask = cv2.erode(sliceMask, kernel)

        if bshow:
            cv2.namedWindow("imgMask", cv2.WINDOW_NORMAL + cv2.WINDOW_KEEPRATIO)
            cv2.imshow("imgMask", sliceMask)
            cv2.waitKey(0)

        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nThickness, nThickness))
        imgMask = cv2.erode(sliceMask, kernel2)
        epidermis = sliceMask - imgMask

        if bshow:
            cv2.namedWindow("epidermis", cv2.WINDOW_NORMAL + cv2.WINDOW_KEEPRATIO)
            cv2.imshow("epidermis", epidermis)
            cv2.waitKey(0)

        return meanI[0], epidermis, sliceMask

    def DetectPeriphery(self, imagefile, epidermis_contour_out, bSaveProcessing=False):
        imgGray = dbsFunction.dbsimread(imagefile, cv2.IMREAD_GRAYSCALE)

        slicemask = np.zeros((imgGray.shape[0], imgGray.shape[1]), np.uint8)
        cv2.drawContours(slicemask, [epidermis_contour_out], -1, 255, -1)
        meanslice = cv2.mean(imgGray, mask=slicemask)

        ret, dst = cv2.threshold(imgGray, int(meanslice[0]), 255, cv2.THRESH_BINARY_INV)
        innermask = cv2.bitwise_and(src1=dst, src2=slicemask, mask=slicemask)

        morph_size = 5
        kernel = np.ones((morph_size, morph_size), np.uint8)

        innermask = cv2.dilate(innermask, (morph_size, morph_size))
        innermask = cv2.erode(innermask, (morph_size, morph_size))

        inner_contour = dbsFunction.GetLargestContour(innermask)

        imgRGB = dbsFunction.dbsimread(imagefile, cv2.IMREAD_COLOR)
        cv2.drawContours(imgRGB, [epidermis_contour_out], -1, (255, 0, 0), 3)
        cv2.drawContours(imgRGB, [inner_contour], -1, (0, 255, 0), 3)

        if bSaveProcessing:
            res_rgb_file = os.path.join(self.test_path_res,
                                        os.path.splitext(os.path.basename(imagefile))[0] + "_periphery.png")
            dbsFunction.dbsimwrite(res_rgb_file, imgRGB, ext=".png")

        return inner_contour

    def one(self, path_images, bSaveProcessing=True, _gignal_progress=None):

        self.test_path_res = os.path.join(path_images, "_results_")
        if not os.path.exists(self.test_path_res):
            os.mkdir(self.test_path_res)
        
        images_fps = dbsFunction.findfullimagefile(path_images)
        cal_res = []
        start_total = time.perf_counter()

        for iiind, fn in enumerate(images_fps):
            print("\rBegin detection:{}".format(fn), end="")
            start_each = time.perf_counter()
            end_str = '% COMPLETED'
            bar = '\r SLICE PIPELINE %3.1f %%  COMPLETED' % (iiind / len(images_fps) * 100)
            print(bar)

            phen_table = micro_pheno_tables(fn, self.test_path_res)
            onename = dbsFunction.getOnlyName(fn)

            # new added! 20210608
            # 假设文件名字前两位
            if False:
                strOuts = re.split('[_.-]', onename)  # 多个字符分割串
                if len(strOuts) >= 2:
                     phen_table.cultivarName = strOuts[0]
                     phen_table.id_internode = int(strOuts[1])
            # 假设文件名字前4位
            if True:
                strOuts = re.split('[_.-]', onename)  # 多个字符分割串

                if len(strOuts) >= 4:
                    phen_table.cultivarName = strOuts[0]
                    phen_table.plant_density = strOuts[2]
                    phen_table.plant_fertilizer = strOuts[1]

                    if strOuts[3] == '':
                        phen_table.id_plant = (strOuts[4])
                    else:
                        phen_table.id_plant = (strOuts[3])

            meanI, epidermis, sliceMask = self.DetectEpidermis(fn, 15, False)
            output_epidermisfile = os.path.join(self.test_path_res, os.path.splitext(os.path.basename(fn))[0] + "_epidermis.png")
            output_slicemaskfile = os.path.join(self.test_path_res, os.path.splitext(os.path.basename(fn))[0] + "_slicemask.png")
            if bSaveProcessing:
                dbsFunction.dbsimwrite(output_epidermisfile, epidermis, ".jpg")
                dbsFunction.dbsimwrite(output_slicemaskfile, sliceMask,  ".png")

            if _gignal_progress is not None:
                str = "->handle {} images in {}".format(iiind+1, len(images_fps))
                npr = 30 + (100-30) / len(images_fps)
                _gignal_progress.emit(int(npr), 100, str, "")

            phen_table.outerCon_epidermis = dbsFunction.GetLargestContour(sliceMask)

            phen_table.inner_contour = self.DetectPeriphery(fn, phen_table.outerCon_epidermis, bSaveProcessing)

            image = dbsFunction.dbsimread(fn, cv2.IMREAD_COLOR)

            origin_size = image.shape

            image = cv2.resize(image, (2080, 2080))

            img, masks, dict_pred_res = vbs_maizestem.predictOneImage(image)

            mask = masks[0]
            mask = cv2.resize(mask, (origin_size[0], origin_size[1]))

            if bSaveProcessing:
                res_file = os.path.join(self.test_path_res, os.path.splitext(os.path.basename(fn))[0] + ".png")
                dbsFunction.dbsimwrite(res_file, mask,  ext=".png")
                print("save segmented mask:", res_file)

            #################################current dir############################################################
            rgbVis = dbsFunction.dbsimread(fn, cv2.IMREAD_COLOR)
            mask2 = mask.astype(np.uint8)
            ######################################################
            bGenerateVisImage = False
            dict_cons, list_cons_discard, mask, rgbVis_split = dbsAdaptiveWatershed.adaptiveWatershed(
                rgbVis,
                mask2,
                minArea = 20,
                ratio_area_threshold = 0.9,
                bGenerateVisImage = bGenerateVisImage
            )

            # 20210628 定义新的维管束分布模式指标：
            cons_ini = dict_cons["cons_ini"]
            cons_individual = dict_cons["cons_individual"]
            cons_multiple = dict_cons["cons_multiple"]
            list_cons_splitted = dict_cons["list_cons_splitted"]
            cons_results = dict_cons["cons_results"]

            # 粘连面积与整体面积比值
            area_ini = dbsFunction.GetTotalArea(cons_ini)
            area_individual = dbsFunction.GetTotalArea(cons_individual)
            area_multiple = dbsFunction.GetTotalArea(cons_multiple)
            area_splitted = 0
            num_splitted = 0
            for cons in list_cons_splitted:
                area_splitted += dbsFunction.GetTotalArea(cons)
                num_splitted += len(cons)
            area_results = dbsFunction.GetTotalArea(cons_results)

            # 20210608 ...
            phen_table.ARIVB = 0
            phen_table.SRVB = 0

            if area_ini > 0:
                phen_table.ARIVB = area_individual / area_ini

            if len(cons_multiple) > 0:
                phen_table.SRVB = num_splitted / len(cons_multiple)

            if bGenerateVisImage:
                res_file = os.path.join(self.test_path_res, os.path.splitext(os.path.basename(fn))[0] + "_watershed.png")
                dbsFunction.dbsimwrite(res_file, mask,  ext=".png")

                res_file = os.path.join(self.test_path_res, os.path.splitext(os.path.basename(fn))[0] + "_watershed_rgb.png")
                dbsFunction.dbsimwrite(res_file, rgbVis_split,  ext=".png")

                res_rgb_file = os.path.join(self.test_path_res, os.path.splitext(os.path.basename(fn))[0] + "_vbs_split.png")
                if bSaveProcessing:
                    dbsFunction.dbsimwrite(res_rgb_file, rgbVis_split, ext=".png")

                self.visImage_layout_1(rgbVis_split,
                                       title="vascular bundles splitting[{}]".format(onename),
                                       visX=0, visY=0, progress=(iiind+1)/ len(images_fps))

            if cv2.__version__ <= '3.4.4':
                image0, contours, hierarchy_vbs = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            else:
                contours, hierarchy_vbs = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            #重新计算 维管束
            if cv2.__version__ <= '3.4.4':
                image0, contours, hierarchy_vbs = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            else:
                contours, hierarchy_vbs = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            contours_filter = []
            contours_omit = []
            for con in contours:
                x,y,wid,hei = cv2.boundingRect(con)

                cpx = x + int(wid/2)
                cpy = y + int(hei/2)

                dis = cv2.pointPolygonTest(phen_table.outerCon_epidermis, (cpx, cpy), True)
                if dis <=0:
                    contours_omit.append(con)
                    pass
                else:
                    contours_filter.append(con)
            contours = []
            contours.extend(contours_filter)

            phen_table.vbs_contours = []
            for ind in range(len(contours)):
                if hierarchy_vbs[0][ind][3] == -1:
                    a = cv2.contourArea(contours[ind])
                    if a > 10:
                        phen_table.vbs_contours.append(contours[ind])


            contours_vbs_outer = []
            for ind in range(len(contours)):
                if len(contours[ind]) < 5:
                    continue
                cv2.drawContours(rgbVis, contours, ind, dbsFunction.getColor(ind), 3)
                if hierarchy_vbs[0][ind][3] == -1:
                    a = cv2.contourArea(contours[ind])
                    if a > 10:
                        contours_vbs_outer.append(contours[ind])

            image_input = dbsFunction.dbsimread(fn, cv2.IMREAD_GRAYSCALE)
            image_input2 = dbsFunction.dbsimread(fn, cv2.IMREAD_GRAYSCALE)
            cv2.drawContours(image_input, contours, -1, 0, -1)

            phen_table.pheno_slice(phen_table.outerCon_epidermis, fn)
            phen_table.EZ_T_Fit = 0
            phen_table.estimate_epidermis_inner = []
            if True:
                maskInn = np.zeros((image_input.shape[0], image_input.shape[1]), np.uint8)
                cons_temp = []
                cons_temp.append(phen_table.outerCon_epidermis)
                cv2.drawContours(maskInn, cons_temp,-1,255,-1)

                cons_layers = []
                cons_layers.append(phen_table.outerCon_epidermis)
                iter_times = 20
                for nThick in range (1, iter_times):
                    masktemp = np.zeros((image_input.shape[0], image_input.shape[1]), np.uint8)
                    cons_temp = []
                    cons_temp.append(phen_table.outerCon_epidermis)
                    cv2.drawContours(masktemp, cons_temp, -1, 255, -1)

                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nThick*2+1, nThick*2+1)) #(nThick*2+1, nThick*2+1))
                    masktemp = cv2.erode(masktemp, kernel)

                    con_temp = dbsFunction.GetLargestContour(masktemp)
                    cons_layers.append(con_temp)


                mean_slice = []
                mean_slice2 = []
                thick_erode = []
                thick_estimate = []
                intensity_estimate = []
                intensity_estimate2 = []
                for nLayer in range(len(cons_layers)-1):
                    con_out = cons_layers[nLayer]
                    con_in = cons_layers[nLayer+1]

                    if len(con_in) <= 0:
                        continue


                    a1= 0
                    a2= 0
                    if len(phen_table.outerCon_epidermis)>0:
                        a1 = cv2.contourArea(phen_table.outerCon_epidermis)
                    if len(con_in) > 0:
                        a2 = cv2.contourArea(con_in)
                    area_temp = a1 - a2 # cv2.contourArea(phen_table.outerCon_epidermis) - cv2.contourArea(con_in)


                    l1= 0
                    l2= 0
                    if len(phen_table.outerCon_epidermis)>0:
                        l1 = cv2.arcLength(phen_table.outerCon_epidermis,True)
                    if len(con_in) > 0:
                        l2 = cv2.arcLength(con_in,True)

                    len_temp = (l1+l2)/2
                    #len_temp = (cv2.arcLength(phen_table.outerCon_epidermis,True)+cv2.arcLength(con_in,True))/2

                    thick_temp = area_temp/len_temp
                    thick_estimate.append(thick_temp)

                    thick_erode.append(nLayer+1)

                    #新增加
                    if True:
                        maskepicermis = np.zeros((image_input.shape[0], image_input.shape[1]), np.uint8)
                        cons_temp = []
                        cons_temp.append(phen_table.outerCon_epidermis)
                        cv2.drawContours(maskepicermis, cons_temp, -1, 255, -1)
                        cons_temp = []
                        cons_temp.append(con_in)
                        cv2.drawContours(maskepicermis, cons_temp, -1, 0, -1)
                        mean_temp = cv2.mean(image_input, mask=maskepicermis)
                        intensity_estimate.append(mean_temp[0])
                    if True:
                        #
                        maskepicermis = np.zeros((image_input.shape[0], image_input.shape[1]), np.uint8)
                        cons_temp = []
                        cons_temp.append(con_out)
                        cv2.drawContours(maskepicermis, cons_temp, -1, 255, -1)
                        cons_temp = []
                        cons_temp.append(con_in)
                        cv2.drawContours(maskepicermis, cons_temp, -1, 0, -1)
                        mean_temp = cv2.mean(image_input, mask=maskepicermis)
                        intensity_estimate2.append(mean_temp[0])


                    cv2.drawContours(maskepicermis, cons_temp, -1, 255, -1)
                    meanInn, stdInn = cv2.meanStdDev(image_input, mask=maskepicermis)

                    meanInn2, stdInn2 = cv2.meanStdDev(image_input2, mask=maskepicermis)

                    mean_slice.append(meanInn[0])
                    mean_slice2.append(meanInn2[0])

                fitX=0
                fitY=0
                for nID in range(1, len(intensity_estimate)-1):
                    if len(intensity_estimate) <=3:
                        continue

                    val1 = intensity_estimate2[nID]-intensity_estimate[nID]
                    val2 = intensity_estimate2[nID+1] - intensity_estimate[nID+1]
                    if (val1 >= 0 and val2 <= 0) :
                        fitX, fitY = dbsFunction.findIntersection(
                            thick_estimate[nID],intensity_estimate[nID],
                            thick_estimate[nID+1], intensity_estimate[nID+1],
                            thick_estimate[nID],intensity_estimate2[nID],
                            thick_estimate[nID+1], intensity_estimate2[nID+1]
                        )

                        phen_table.EZ_T_Fit = fitX

                        phen_table.estimate_epidermis_inner = cons_layers[nID]

                        break

                    # note:
                    if len(phen_table.estimate_epidermis_inner) <= 0:
                        phen_table.estimate_epidermis_inner = cons_layers[1]

                # draw plot estimate thickness
                if True:
                    fig = plt.figure(figsize=(6, 3))
                    plt.yticks(fontproperties='Calibri')  # , size=15, weight='bold')  # 设置大小及加粗
                    plt.xticks(fontproperties='Calibri')  # , size=15)
                    plt.rcParams['font.sans-serif'] = ['Calibri']  # 用来正常显示中文标签，如果想要用新罗马字体，改成 Times New Roman
                    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
                    plt.plot(thick_estimate, intensity_estimate, label='AIEP') # Average intensity of estimated epidermis (AIEP)
                    plt.plot(thick_estimate, intensity_estimate2, label='AINR') # Average intensity of new region
                    plt.scatter([fitX], [fitY], 12, 'r')  # 绘制点x,y,大小，颜色
                    plt.plot([fitX, fitX], [0, fitY], 'b--') # 合适的点画线
                    plt.annotate('Thickness:%.2f'%(fitX), xy=(fitX, fitY), xytext=(fitX+1, fitY+15),
                                 arrowprops=dict(facecolor='k', shrink=0.02)#, headlength = 5, headwidth = 10, width = 10)#)
                    )  # 注释的地方xy(x,y)和插入文本的地方xytext(x1,y1)

                    plt.xlim((0, max(thick_estimate)+1))
                    plt.ylim((0, max(intensity_estimate2)+5))

                    plt.xlabel('Erodes of slice mask (pixels)') #'Distance from the outmost boundary of slice (pixels)')
                    plt.ylabel('Average intensity') # Pixel mean intensity')
                    plt.title('Estimate epidermis thickness')

                    plt.tight_layout()  # 解决绘图时上下标题重叠现象
                    plt.legend()
                    res_rgb_file = os.path.join(self.test_path_res,
                                                os.path.splitext(os.path.basename(fn))[0] + "_thickness.jpg")
                    plt.savefig(res_rgb_file, dpi=300)
                    #print("output plot file:", res_rgb_file)
                    plt.close(fig)
                #
                cv2.drawContours(rgbVis, [phen_table.estimate_epidermis_inner], -1, (255,255,255), 3)
                cv2.drawContours(rgbVis, [phen_table.outerCon_epidermis], -1, (255,255,255), 3)

                res_rgb_file = os.path.join(self.test_path_res, os.path.splitext(os.path.basename(fn))[0] + "_rgb.png")
                # cv2.imwrite(res_rgb_file, rgbVis)
                if bSaveProcessing:
                    dbsFunction.dbsimwrite(res_rgb_file, rgbVis, ext=".png")

            #---------------------------------------------------
            # 计算表皮区域
            phen_table.pheno_epidermis(epidermis_contour_out=phen_table.outerCon_epidermis,
                                       epidermis_contour_in=phen_table.estimate_epidermis_inner,
                                       ctfilename=fn)
            end_each = time.perf_counter()
            totaltime_each = (end_each - start_each)
            phen_table.caltime = "%.3f"%(totaltime_each)

            # output
            phen_table.pheno_periphery_inner(
                ctfilename = fn,
                bSaveProcessing = bSaveProcessing
            )

            # output...
            phen_table.pheno_vbs(phen_table.vbs_contours, fn, bSaveProcessing)

            # 20210908
            phen_table.pheno_layers(phen_table.vbs_contours,
                                    epidermis_contour_out=phen_table.outerCon_epidermis,
                                    epidermis_contour_in=phen_table.estimate_epidermis_inner,
                                    ctfilename = fn
                                    )

            # output
            cal_res.append(phen_table)

        end_total = time.perf_counter()
        basepath, filename = os.path.split(fn)
        lastpath, typename = os.path.split(basepath)

        print("Completed ({} images) ".format(images_fps))
        #########################################
        res_file = os.path.join(self.test_path_res, typename + "_statistic.csv")
        micro_pheno_tables.write_serialize_statistic(res_file, cal_res)
        #########################################
        # 20200908 输出维管束力学指标
        res_file_mechanics = os.path.join(self.test_path_res, typename + "_statistic_mechanics.csv")
        micro_pheno_tables.write_serialize_statistic_mechanics(res_file_mechanics, cal_res)
        print("Output stat csv:", res_file)
        dict_output = {}
        dict_output["stat results"] = res_file
        #########################################
        # 2020
        # 显示执行进度
        totaltime = (end_total - start_total)
        print("Exteract pots has total run time: %.03f seconds" % (
                    end_total - start_total))

        dict_output["totaltime"] = totaltime

        time_res_file = os.path.join(basepath, typename + "time_statistic.csv")
        with open(time_res_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            tags = []
            tags.append("Total cal time(ms)")
            tags.append("%.f" % totaltime)
            writer.writerow(tags)
            tags = []
            tags.append("Total samples")
            tags.append("%d" % len(images_fps))
            writer.writerow(tags)  # 写入1行用writerow; row_data是你要写入的数据，最好是list类型。
            #print(tags)

            tags = []
            tags.append("Average cal time(s)")
            tags.append("%.f" % (totaltime / len(images_fps)))
            writer.writerow(tags)  # 写入1行用writerow; row_data是你要写入的数据，最好是list类型。
            #print(tags)

        ###################
        # plot
        #path_phenotyping = os.path.join(self.test_path_res, r"phenotyping")
        #if not os.path.exists(self.path_phenotyping):
        #    os.mkdir(self.path_phenotyping)
        #micro_pheno_tables.serialimage_pheno_plot_vis(path_phenotyping, cal_res)

        #########################
        ## save
        #name_path = dbsFunction.getOnlyName(path_images)
        #pkl_file = os.path.join(path_images, "{}.pkl".format(name_path))
        #micro_pheno_tables.save2pkl(pkl_file, cal_res)
        #print("write micro_pheno_tables to: {}".format(pkl_file))

    def predictOneImage(img):
        '''
        example:
        # imgfile = r"E:\++deTasseling++\data\_dataset_mmseg_\leaf_vb_todo\leaf_data1_22.2.18\36-1-1__rec_Sng_.bmp"
        '''
        palette = 'cityscapes'
        opacity = 0.5

        start = time.perf_counter()
        # test a single image
        results = inference_segmentor(model, img)
        dict_pred_res = {}

        masks = []
        num_class = len(results)
        for i in range(num_class):
            id = i + 1
            result = results[i]
            result[result == id] = 255
            result = np.uint8(result)
            masks.append(result)

            contours, hierarchy = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            index = 0
            list_vbs = []
            areaTotal = 0
            areaAverage = 0
            for ii in range(0, len(contours)):
                x, y, w, h = cv2.boundingRect(contours[ii])
                area = cv2.contourArea(contours[ii])
                if area < 8:
                    continue
                list_vbs.append(contours[ii])
                areaTotal += area

                span = 5
                cv2.rectangle(img, (x - span, y - span), (x + w + span * 2, y + w + span * 2), (0, 0, 255), 3)

            if len(list_vbs) > 0:
                areaAverage = areaTotal / len(list_vbs)
            dict_pred_res["vbsNum"] = len(list_vbs)
            dict_pred_res["areaTotal"] = areaTotal
            dict_pred_res["areaAverage"] = areaAverage

            cv2.putText(img, "NUM: %d" % (len(list_vbs)),
                        (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.putText(img, "Total Area:%.3f" % (areaTotal),
                        (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.putText(img, "Average Area:%.3f" % (areaAverage),
                        (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            #print("dict_pred_res:", dict_pred_res)
        return img, masks, dict_pred_res

if __name__ == "__main__":
    path_images = r"vbs_images\maizestem"
    vv = vbs_maizestem()
    vv.one(path_images)
    