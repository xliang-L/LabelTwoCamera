# 需求： 我需要将matchDATA中的图片 打bbox标签。
# 标签文件： 1. 盘片编号 2. 传感器编号 3. bbox坐标 4. 类别
# 生成一个类似于labelimg的标注程序,但是能够同时显示dv和dvs的图片，并且能够同时标注。
# 使用opencv显示图片，并且能够同时标注。
# 使用labelimg的标注格式。

# 盘片编号 是 图片路径中 E:\splitDATA\matchDATA\旋转\1 中的1
# 传感器编号 是 图片路径中 E:\splitDATA\matchDATA\旋转\1\dv 或 E:\splitDATA\matchDATA\旋转\1\dvs 中的dv 或 dvs

import cv2
import numpy as np
import os
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from tkinter import simpledialog, Tk, filedialog, messagebox

class LabelTool:
    def __init__(self):
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.current_bbox = []
        self.all_bboxes = []
        self.current_class = 1  # 当前类别
        self.scale = 0.5  # 添加缩放比例
        self.dv_circle_bbox = None  # dv传感器的圆形区域
        self.dvs_circle_bbox = None  # dvs传感器的圆形区域
        self.scale_x,self.scale_y = -1,-1
        self.imgx,self.imgy = -1,-1
        # 数据目录和图像列表
        self.root = Tk()
        self.root.withdraw()
        self.data_list = []
        self.data_dir = self.select_data_dir()
        if not self.data_dir:
            raise Exception("未选择数据目录")
        
        self.dv_images = []
        self.dvs_images = []
        self.current_index = 0
        self.load_image_list()
        
        # 图像和窗口设置
        self.dv_window = "DV Image"
        self.dvs_window = "DVS Image"
        self.current_dv_img = None
        self.current_dvs_img = None
        
        # 标注数据
        self.labels = {
            "disk_id": self.get_disk_id(),
            "bboxes": []
        }
    
    def get_disk_id(self):
        return os.path.basename(self.data_dir)
    
    def get_sensor_id(self, sensor_type):
        return sensor_type

    def load_image_list(self):
        # 加载DV和DVS图像列表
        self.dv_images = sorted([f for f in os.listdir(os.path.join(self.data_dir, 'dv')) 
                               if f.endswith(('.jpg', '.png'))])
        self.dvs_images = sorted([f for f in os.listdir(os.path.join(self.data_dir, 'dvs')) 
                                if f.endswith(('.jpg', '.png'))])
        if not self.dv_images or not self.dvs_images:
            raise Exception("未找到图像文件")
    
    def load_current_images(self):
        # 加载当前索引的图像对
        dv_path = os.path.join(self.data_dir, 'dv', self.dv_images[self.current_index])
        dvs_path = os.path.join(self.data_dir, 'dvs', self.dvs_images[self.current_index])
        self.current_dv_img = cv2.imread(dv_path)
        self.current_dvs_img = cv2.imread(dvs_path)
        
        # 显示当前图像信息
        # self.crop_images()
        self.imgx,self.imgy = self.current_dv_img.shape[1],self.current_dv_img.shape[0]
        print(f"当前图像: {self.current_index + 1}/{len(self.dv_images)}")
        
        
    def labelCircle(self):
        # 标注圆形区域
        scaled_dv_img = cv2.resize(self.current_dv_img, (0,0), fx=0.3, fy=0.3)  # 将DV图像缩小为0.3
        cv2.imshow(self.dv_window, scaled_dv_img)
        print("请在DV图像上标注圆形区域")
        self.dv_circle_bbox = cv2.selectROI(self.dv_window, scaled_dv_img, fromCenter=False, showCrosshair=True)
        
        cv2.imshow(self.dvs_window, self.current_dvs_img)
        print("请在DVS图像上标注圆形区域")
        self.dvs_circle_bbox = cv2.selectROI(self.dvs_window, self.current_dvs_img, fromCenter=False, showCrosshair=True)
        
        # 将标注框的坐标转换为原图的坐标
        self.dv_circle_bbox = (int(self.dv_circle_bbox[0] / 0.3), int(self.dv_circle_bbox[1] / 0.3), 
                               int(self.dv_circle_bbox[2] / 0.3), int(self.dv_circle_bbox[3] / 0.3))
        
        print(f"圆形区域标注完成：{self.dv_circle_bbox}, {self.dvs_circle_bbox}")

    def crop_images(self):
        # 根据标定的圆裁剪图像
        if self.dv_circle_bbox is not None:
            x, y, w, h = self.dv_circle_bbox
            self.current_dv_img = self.current_dv_img[y:y+h, x:x+w]
        if self.dvs_circle_bbox is not None:
            x, y, w, h = self.dvs_circle_bbox
            self.current_dvs_img = self.current_dvs_img[y:y+h, x:x+w]

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                img_copy = self.display_scaled_image(draw = True)
                cv2.rectangle(img_copy, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
                cv2.imshow(self.dv_window, img_copy)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = min(self.ix, x), min(self.iy, y)
            x2, y2 = max(self.ix, x), max(self.iy, y)
            # 考虑缩放比例
            if self.scale == 0.5:
                
                x1 = int(x1 / self.scale)
                y1 = int(y1 / self.scale)
                x2 = int(x2 / self.scale)
                y2 = int(y2 / self.scale)
            elif self.scale == 1:
                x1 = int(x1)+self.scale_x - self.imgx//4
                y1 = int(y1)+self.scale_y - self.imgy//4
                x2 = int(x2)+self.scale_x - self.imgx//4
                y2 = int(y2)+self.scale_y - self.imgy//4
            # 过滤鼠标误点，确保框的宽高大于0
            if x1 < x2 and y1 < y2:
                bbox_dict = {
                    "disk_id": self.get_disk_id(),
                    "sensor_id": self.get_sensor_id('dv'),  # dv窗口的标注
                    "bbox": [x1, y1, x2, y2],
                    "class": self.current_class
                }
                self.all_bboxes.append(bbox_dict)  # 将字典添加到所有标注中
                
                img_copy = self.current_dv_img.copy()
                self.draw_all_bboxes(img_copy,'dv')
                self.display_scaled_image()
                # 自动保存标注
                self.save_labels()

        elif event == cv2.EVENT_MOUSEWHEEL:  # 处理鼠标滚轮事件
            # 向上滚动，按照scale = 1 
            # 向下滚动，按照scale = 0.5
            if flags > 0:
                if self.scale == 1:
                    self.scale_x = x
                    self.scale_y = y
                else:
                    self.scale_x =2* x
                    self.scale_y =2* y    
                    self.scale = 1
            else:

                self.scale = 0.5
            self.display_scaled_image()

    def mouse_callback_dvs(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                img_copy = self.current_dvs_img.copy()
                self.draw_all_bboxes(img_copy,'dvs')
                cv2.rectangle(img_copy, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
                cv2.imshow(self.dvs_window, img_copy)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = min(self.ix, x), min(self.iy, y)
            x2, y2 = max(self.ix, x), max(self.iy, y)
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            # 过滤鼠标误点，确保框的宽高大于0
            if x1 < x2 and y1 < y2:
                bbox_dict = {
                    "disk_id": self.get_disk_id(),
                    "sensor_id": self.get_sensor_id('dvs'),  # dvs窗口的标注
                    "bbox": [x1, y1, x2, y2],
                    "class": self.current_class
                }
                self.all_bboxes.append(bbox_dict)  # 将字典添加到所有标注中
                
                img_copy = self.current_dv_img.copy()
                self.draw_all_bboxes(img_copy,'dvs')

                # 自动保存标注
                self.save_labels()

    def display_scaled_image(self,draw = False):
        if self.scale == 1:
            # 以原始尺寸显示半张图片，即图片中心 为 self.scale_x,self.scale_y, 向左向右向上向下分别截取imgx/4,imgy/4
            if self.current_dv_img is not None and not draw:
                self.draw_all_bboxes(self.current_dv_img,'dv')
                # 初始化scaled_img为一张纯黑的图片
                scaled_img = np.zeros((self.imgy//2, self.imgx//2, 3), dtype=np.uint8)

                # 判断是不是会超出图片边界，如果超出部分设置为0
                if self.scale_x - self.imgx//4 < 0:
                    self.scale_x = self.imgx//4
                if self.scale_x + self.imgx//4 > self.imgx:
                    self.scale_x = self.imgx - self.imgx//4
                if self.scale_y - self.imgy//4 < 0:
                    self.scale_y = self.imgy//4
                if self.scale_y + self.imgy//4 > self.imgy:
                    self.scale_y = self.imgy - self.imgy//4

                scaled_img = self.current_dv_img[self.scale_y - self.imgy//4:self.scale_y + self.imgy//4,self.scale_x - self.imgx//4:self.scale_x + self.imgx//4]

                cv2.imshow(self.dv_window, scaled_img)
            if not draw:
                if self.current_dvs_img is not None:
                    self.draw_all_bboxes(self.current_dvs_img,'dvs')
                    scaled_img = self.current_dvs_img[self.scale_y - self.imgy//4:self.scale_y + self.imgy//4,self.scale_x - self.imgx//4:self.scale_x + self.imgx//4]
                    return scaled_img
                
        elif self.scale == 0.5:
            if self.current_dv_img is not None and not draw:
                self.draw_all_bboxes(self.current_dv_img,'dv')
                scaled_img = cv2.resize(self.current_dv_img, 
                                        (0, 0), 
                                        fx=self.scale, 
                                        fy=self.scale)
                
                
                cv2.imshow(self.dv_window, scaled_img)

            if not draw:
                if self.current_dvs_img is not None:
                    self.draw_all_bboxes(self.current_dvs_img,'dvs')
                    scaled_img = cv2.resize(self.current_dvs_img, 
                                            (0, 0), 
                                            fx=self.scale, 
                                            fy=self.scale)
                    return scaled_img

    def draw_all_bboxes(self, img,sensor_id):
        for bbox in self.all_bboxes:
            if bbox["sensor_id"] == sensor_id:
                # 类别1为绿色，2为蓝色，3为红色，4为黄色
                color = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)][bbox["class"] - 1]
                cv2.rectangle(img, (bbox["bbox"][0], bbox["bbox"][1]), 
                            (bbox["bbox"][2], bbox["bbox"][3]), color, 1)  # 将线宽改为1
                # 类别1为划痕,类别2为点痕,类别3为异色亮带,类别四为污渍.
                txt = ['Scratch', 'point', 'band', 'stains'][bbox["class"] - 1]
                cv2.putText(img, txt, (bbox["bbox"][0], bbox["bbox"][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)  # 将字体大小和线宽改为更细

    def select_data_dir(self):
        """选择数据目录"""
        data_dir = filedialog.askdirectory(title="选择数据目录")
        print(f"当前标注的数据目录: {data_dir}")

        # data_dir下是数个文件夹，我要找到这里面哪个文件夹里含有dv和dvs 文件夹
        

        # 遍历data_dir下的所有文件夹
        for folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder).replace('\\','/')
            # 检查是否为文件夹
            if os.path.isdir(folder_path):
                # 检查dv和dvs文件夹是否存在
                if os.path.exists(os.path.join(folder_path, 'dv')) and os.path.exists(os.path.join(folder_path, 'dvs')):
                    print(f"找到包含'dv'和'dvs'文件夹的目录: {folder_path}")
                    self.data_list.append(folder_path)  # 保存找到的目录
                
        if len(self.data_list) == 0:
            print("警告：在所选目录中未找到包含'dv'和'dvs'文件夹的子文件夹。")
        return self.data_list.pop() if len(self.data_list) != 0 else None

    
    def get_user_confirmation(self):
        """获取用户确认"""
        return messagebox.askyesno("确认", 
            "所选目录结构可能不正确。是否继续？\n" +
            "正确的目录结构应该包含：\n" +
            " - dv/\n" +
            " - dvs/")

    def create_xml_annotation(self, filename):
        root = ET.Element("annotation")
        
        # 添加基本信息
        ET.SubElement(root, "filename").text = filename
        ET.SubElement(root, "disk_id").text = self.labels["disk_id"]
        
        # 添加图像尺寸信息
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(self.current_dv_img.shape[1])
        ET.SubElement(size, "height").text = str(self.current_dv_img.shape[0])
        ET.SubElement(size, "depth").text = str(self.current_dv_img.shape[2])
        
        # 添加所有bbox
        for bbox in self.all_bboxes:
            obj = ET.SubElement(root, "object")
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(bbox["bbox"][0])
            ET.SubElement(bndbox, "ymin").text = str(bbox["bbox"][1])
            ET.SubElement(bndbox, "xmax").text = str(bbox["bbox"][2])
            ET.SubElement(bndbox, "ymax").text = str(bbox["bbox"][3])
            ET.SubElement(obj, "class").text = str(bbox["class"])  # 添加类别信息
        
        return ET.ElementTree(root)
    
    def save_labels(self):
        # 创建label文件夹
        # label_dir 在data_dir 上一层建立一个label文件夹，里面建立一个和data_dir同名的文件夹

        label_dir = os.path.join(os.path.dirname(self.data_dir), 'label', os.path.basename(self.data_dir))

        os.makedirs(label_dir, exist_ok=True)

        # 保存JSON格式ccc
        json_filename = os.path.join(label_dir, f"{self.dv_images[self.current_index][:-4]}.json")
        with open(json_filename, 'w') as f:
            json.dump({"disk_id": self.labels["disk_id"], "bboxes": self.all_bboxes}, f, indent=4)
            
        # 保存XML格式
        xml_filename = os.path.join(label_dir, f"{self.dv_images[self.current_index][:-4]}.xml")
        xml_tree = self.create_xml_annotation(self.dv_images[self.current_index])
        xml_tree.write(xml_filename, encoding='utf-8', xml_declaration=True)
        
        print(f"标注已保存至 {json_filename} 和 {xml_filename}")

    def run(self):
        cv2.namedWindow(self.dv_window)
        cv2.namedWindow(self.dvs_window)



   
        cv2.setMouseCallback(self.dv_window, self.mouse_callback)
        cv2.setMouseCallback(self.dvs_window, self.mouse_callback_dvs)  # 添加dvs窗口的鼠标回调
        
        self.load_current_images()

        # self.labelCircle()  # 先标定圆的位置/

        # # # self.crop_images()  # 裁剪图像
        # # # self.load_current_images()
        # self.load_current_images()
        while True:
            # print("按 's' 保存标注")
            
            if self.current_dv_img is not None:
                self.display_scaled_image()
            if self.current_dvs_img is not None:

                img_copy_dvs = self.current_dvs_img.copy()
                self.draw_all_bboxes(img_copy_dvs,'dvs')
                cv2.imshow(self.dvs_window, img_copy_dvs)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC键退出
                break
            elif key == ord('s'):  # 's'键保存标注
                self.save_labels()
            elif key == ord('d'):  # 'd'键下一张
                if self.current_index < len(self.dv_images) - 1:
                    self.current_index += 1
                    self.all_bboxes = []  # 清空当前标注
                    self.load_current_images()
                    # self.crop_images()  # 裁剪图像
            elif key == ord('a'):  # 'a'键上一张
                if self.current_index > 0:
                    self.current_index -= 1
                    self.all_bboxes = []  # 清空当前标注
                    self.load_current_images()
                    # self.crop_images()  # 裁剪图像
            elif key == ord('c'):  # 'c'键清除上一个标注
                if self.all_bboxes:
                    self.all_bboxes.pop()  # 移除最后一个标注
                    self.save_labels()
                    self.load_current_images()
            elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:  # 按下数字键选择类别
                self.current_class = key - ord('0')  # 设置当前类别为1, 2, 3, 4
            elif key == ord('g'):
                self.labelCircle()
            elif key == ord('q'):
                if self.data_list==[]:
                    break
                else:
                    self.data_dir = self.data_list.pop()
                    self.load_image_list()
                    self.current_index = 0
                    self.load_current_images()
                    self.all_bboxes = []
                    self.labels = {
                        "disk_id": self.get_disk_id(),
                        "bboxes": []
                    }
                    # self.labelCircle()
                
            
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    tool = LabelTool()
    tool.run()