import numpy as np
from osgeo import gdal
import os
import math
from datetime import datetime
import xml.etree.ElementTree as ET
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import re
from scipy import ndimage



def writeTif(output_path, projection, geotransform, data, dtype):
    gtiff_driver = gdal.GetDriverByName('GTiff')
    if len(data.shape) == 2:
        band_num = 1
        row = data.shape[0]
        column = data.shape[1]
        print(row, column)
    elif len(data.shape) == 3:
        band_num = len(data)
        row = data.shape[1]
        column = data.shape[2]
        print(row, column)
    out_ds = gtiff_driver.Create(output_path, column, row, band_num, dtype)
    out_ds.SetProjection(projection)
    out_ds.SetGeoTransform(geotransform)
    if band_num == 1:
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(data)
        del out_ds
    elif band_num > 1:
        for r in range(len(data)):
            out_band = out_ds.GetRasterBand(r + 1)
            out_band.WriteArray(data[r])
        del out_ds


def get_obsGeometry(path):
    # 解析XML文件
    tree = ET.parse(path)  # 将 'metadata.xml' 替换为你的文件路径
    root = tree.getroot()

    # 查找SolarZenith标签
    solar_zenith_element = root.find('.//SolarZenith')  # 使用XPath查找任意层级的SolarZenith元素
    satellite_zenith_element = root.find('.//SatelliteZenith')  # 使用XPath查找任意层级的SolarZenith元素
    solar_azimuth_element = root.find('.//SolarAzimuth')  # 太阳方位角
    satellite_azimuth_element = root.find('.//SatelliteAzimuth')  # 卫星方位角

    # 提取并打印太阳天顶角数值
    solar_zenith = float(solar_zenith_element.text)
    sate_zenith = float(satellite_zenith_element.text)
    solar_azimuth = float(solar_azimuth_element.text)
    sate_azimuth = float(satellite_azimuth_element.text)
    return solar_zenith, 90 - sate_zenith, solar_azimuth, sate_azimuth


# 计算水体分类影像中每个类别的最小反射率，取最小的10%，然后取平均
def Ca_Cloudfree_Refle(water_class, b1_value, b2_value, b3_value, b4_value):
    # 获取分类标签的所有值，并去除255
    labels = list(np.unique(water_class))
    labels.remove(255)
    # 将计算结果存到labels_mean
    labels_mean_b1 = []
    labels_mean_b2 = []
    labels_mean_b3 = []
    labels_mean_b4 = []

    # 获取最小的20%的值，然后取中位数作为labels_mean_b1
    for k in labels:
        value = k
        test_data = np.array(b1_value[water_class == value])
        num_20 = int(len(test_data) * 0.1)

        # 方法1a：使用np.partition（更高效，不完整排序）
        smallest_20_percent = np.partition(test_data, num_20)[:num_20]
        # print("最小的20%数据（partition方法）:", smallest_20_percent)
        labels_mean_b1.append(np.mean(smallest_20_percent))

    # 不只是要获取b1的平均值，还有b2 b3 b4的平均值
    for k in range(len(labels)):
        # 获取b1_value的范围
        # 计算下限（向下取整到千分位）
        lower_bound = round(labels_mean_b1[k] - 0.001, 3)
        # 计算上限（向上取整到千分位）
        upper_bound = round(labels_mean_b1[k] + 0.001, 3)

        # 获取b1_value中位于范围内的所有波段的数据
        selected_pixels_loc = np.where((water_class == labels[k]) & (b1_value > lower_bound) & (b1_value < upper_bound))
        labels_mean_b2.append(np.mean(b2_value[selected_pixels_loc]))
        labels_mean_b3.append(np.mean(b3_value[selected_pixels_loc]))
        labels_mean_b4.append(np.mean(b4_value[selected_pixels_loc]))
    return labels, labels_mean_b1, labels_mean_b2, labels_mean_b3, labels_mean_b4


def extract_MODTRAN(filename, data_type):  # 从txt文件中提取特定数据，并存储为（162，7，4）的数组，包括T，R和T——slope
    """
    简化版本的数据提取
    """
    # 读取所有行
    with open(filename, 'r') as file:
        lines = file.readlines()

    # 查找所有表头行的位置
    header_lines = []
    for i, line in enumerate(lines):
        if ('altostratus' in line) | ('cirrus' in line):
            header_lines.append(i)
            # print(f"找到表头在第 {i+1} 行")

    print(f"\n总共找到 {len(header_lines)} 个数据模块")
    # 存储结果
    modules = []

    # 提取每个模块
    for i, header_line_num in enumerate(header_lines):
        # print(f"\n提取模块 {i+1} (从第 {header_line_num+1} 行开始)...")

        # 表头行
        # header = lines[header_line_num].strip()
        # print('表头',header)

        # 数据行（从表头下八行开始，直到下一个表头或文件结束），根据数据类型提取不同的数据
        if data_type == 'T':

            data_start = header_line_num + 8
            if i < len(header_lines) - 1:
                data_end = header_lines[i + 1] - 9
            else:
                data_end = len(lines) - 7

            data_lines = lines[data_start:data_end]
            # print(data_lines)
        elif data_type == 'R':
            data_start = header_line_num + 15
            if i < len(header_lines) - 1:
                data_end = header_lines[i + 1] - 2
            else:
                data_end = len(lines)

            data_lines = lines[data_start:data_end]
        elif data_type == 'T_slope':
            data_start = header_line_num + 1
            if i < len(header_lines) - 1:
                data_end = header_lines[i + 1] - 16
            else:
                data_end = len(lines) - 14

            data_lines = lines[data_start:data_end]
            # 处理数据
        data = []
        for line in data_lines:
            line = line.strip()
            if line and not line.startswith('!'):  # 跳过空行和注释行
                # 简单的空格分割
                row_data = line.split()
                if len(row_data) >= 1:
                    data.append(row_data)

        modules.append(data)
        # print(f"模块 {i+1} 提取完成: {len(df)} 行数据")

    return modules


def create_5d_simulation_data(data_extract_MODTRAN):  # 利用从extract_MODTRAN函数中获取的（162，7，4）的数组生成查找表
    """
    创建5维模拟数据
    维度: 太阳天顶角 × 卫星天顶角 × 气溶胶光学厚度 × 云光学厚度 × 波长
    """
    # 创建5维数组
    shape = (len(sun_zenith), len(sat_zenith), len(aerosol_od), len(cloud_od), len(wavelengths))

    # 用有物理意义的模拟数据（您应该替换为实际模拟数据）
    # 这里使用一个简单的模型：反射率随云光学厚度增加而增加，随波长变化
    data = np.zeros(shape)
    num = 0
    for i, sz in enumerate(sun_zenith):
        for j, satz in enumerate(sat_zenith):
            for k, aod in enumerate(aerosol_od):
                # 简化的物理模型（请替换为您的实际数据）

                data[i, j, k, :, :] = data_extract_MODTRAN[num]
                num += 1
    print(data.shape)
    # 创建xarray数据集
    ds = xr.Dataset({
        'reflectance': (['sun_zenith', 'sat_zenith', 'aerosol_od', 'cloud_od', 'wavelength'], data)
    }, coords={
        'sun_zenith': sun_zenith,
        'sat_zenith': sat_zenith,
        'aerosol_od': aerosol_od,
        'cloud_od': cloud_od,
        'wavelength': wavelengths
    })

    # 添加属性描述
    ds.attrs['description'] = '多波长云反射率查找表'
    ds.reflectance.attrs['units'] = '无量纲'
    ds.reflectance.attrs['long_name'] = '云反射率'

    return ds


# 保存查找表
def save_lookup_table(ds, filename):
    ds.to_netcdf(filename)
    print(f"5维查找表已保存到: {filename}")


# 加载查找表
def load_lookup_table(filename):
    return xr.open_dataset(filename)


class SimpleCloudReflectanceLookup:  #
    def __init__(self, lookup_table_path):
        """初始化查找表"""
        self.ds = load_lookup_table(lookup_table_path)
        self._create_interpolator()

    def _create_interpolator(self):
        """创建只在3个维度插值的插值器"""
        # 只在太阳天顶角、卫星天顶角、气溶胶光学厚度这三个维度插值
        points_3d = (
            self.ds.sun_zenith.values,
            self.ds.sat_zenith.values,
            self.ds.aerosol_od.values
        )

        # 创建3维插值器，对每个(云光学厚度, 波长)组合分别插值
        self.interpolators = {}

        for i, cod in enumerate(self.ds.cloud_od.values):
            for j, wl in enumerate(self.ds.wavelength.values):
                # 提取该(云光学厚度, 波长)组合下的3维数据
                data_3d = self.ds.reflectance.values[:, :, :, i, j]
                self.interpolators[(cod, wl)] = RegularGridInterpolator(  # 对于每个(云光学厚度, 波长)组合分别构建插值器
                    points_3d, data_3d, method='linear',
                    bounds_error=False, fill_value=None
                )

    def get_reflectance_matrix(self, sun_zenith, sat_zenith, aerosol_od):
        """
        获取7×4反射率矩阵

        参数:
        - sun_zenith: 太阳天顶角
        - sat_zenith: 卫星天顶角
        - aerosol_od: 气溶胶光学厚度

        返回:
        - 7×4的反射率矩阵
        """
        reflectance_matrix = np.zeros((len(self.ds.cloud_od), len(self.ds.wavelength)))

        for i, cod in enumerate(self.ds.cloud_od.values):
            for j, wl in enumerate(self.ds.wavelength.values):
                # 构建查询点 [太阳天顶角, 卫星天顶角, 气溶胶光学厚度]
                query_point = np.array([sun_zenith, sat_zenith, aerosol_od])

                # 使用对应的插值器计算反射率
                reflectance_matrix[i, j] = self.interpolators[(cod, wl)](
                    query_point).item()  # self.interpolators[(cod, wl)]是一个插值器

        return reflectance_matrix


def parse_envi_hdr(file_path):
    """
    解析ENVI头文件并提取所有参数

    参数:
    file_path: ENVI头文件路径

    返回:
    包含所有参数的字典
    """
    params = {}

    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # 使用正则表达式匹配所有参数
        # 匹配格式: key = value
        pattern = r'([^=\n\r]+)\s*=\s*({[^}]*}|[^\n\r]*)'
        matches = re.findall(pattern, content)

        for key, value in matches:
            key = key.strip().lower()  # 转换为小写以便统一处理

            # 处理大括号内的值
            if value.strip().startswith('{'):
                # 提取大括号内的内容并分割为列表
                inner_content = re.search(r'{(.*)}', value, re.DOTALL)
                if inner_content:
                    # 分割字符串并去除空白
                    values = [v.strip() for v in inner_content.group(1).split(',') if v.strip()]
                    # 尝试转换为浮点数
                    try:
                        params[key] = [float(v) for v in values]
                    except ValueError:
                        params[key] = values
                else:
                    params[key] = value.strip()
            else:
                # 尝试转换为数字
                try:
                    params[key] = float(value.strip())
                except ValueError:
                    params[key] = value.strip()

        return params

    except Exception as e:
        print(f"解析ENVI头文件时出错: {e}")
        return None


def copy_hdr_sections(source_path, target_path, sections_to_copy):
    """
    从源 .hdr 文件复制特定部分到目标 .hdr 文件

    参数:
    source_path: 源 .hdr 文件路径
    target_path: 目标 .hdr 文件路径
    sections_to_copy: 要复制的部分列表，如 ['wavelength', 'fwhm', 'acquisition time']
    """
    # 读取源文件内容
    with open(source_path, 'r') as f:
        source_content = f.readlines()

    # 读取目标文件内容
    with open(target_path, 'r') as f:
        target_content = f.readlines()

    # 提取要复制的部分
    sections_content = []
    copy_section = False
    current_section = None
    brace_count = 0  # 用于跟踪大括号的嵌套

    for line in source_content:
        # 检查是否是要复制的部分的开始
        if not copy_section:
            for section in sections_to_copy:
                if line.strip().startswith(section):
                    copy_section = True
                    current_section = section
                    sections_content.append(line)
                    # 检查是否有大括号开始
                    if '{' in line:
                        brace_count += line.count('{')
                    break
        else:
            # 如果正在复制部分中，继续添加行
            sections_content.append(line)

            # 更新大括号计数
            if '{' in line:
                brace_count += line.count('{')
            if '}' in line:
                brace_count -= line.count('}')

            # 检查是否到达部分结尾
            # 如果没有大括号嵌套，且遇到空行或新部分开始，则结束
            if brace_count == 0 and (line.strip() == '' or
                                     any(line.strip().startswith(s + ' =') for s in sections_to_copy if
                                         s != current_section)):
                copy_section = False
                current_section = None

    # 从目标内容中移除已存在的要复制的部分
    new_target_content = []
    skip_section = False
    target_brace_count = 0

    for line in target_content:
        # 检查是否是要跳过的部分的开始
        if not skip_section:
            skip_current = False
            for section in sections_to_copy:
                if line.strip().startswith(section):
                    skip_section = True
                    skip_current = True
                    # 检查是否有大括号开始
                    if '{' in line:
                        target_brace_count += line.count('{')
                    break

            if not skip_current:
                new_target_content.append(line)
        else:
            # 更新大括号计数
            if '{' in line:
                target_brace_count += line.count('{')
            if '}' in line:
                target_brace_count -= line.count('}')

            # 检查是否到达部分结尾
            if target_brace_count == 0 and (line.strip() == '' or
                                            any(line.strip().startswith(s + ' =') for s in sections_to_copy)):
                skip_section = False
                # 不添加这一行，因为它属于要跳过的部分

    # 将提取的内容添加到目标文件
    with open(target_path, 'w') as f:
        # 写入处理后的目标内容
        f.writelines(new_target_content)

        # 添加新行和提取的内容
        if new_target_content and new_target_content[-1].strip() != '':
            f.write('\n')
        f.writelines(sections_content)


def add_hdr_metadata(hdr_file_path):
    """
    在ENVI .hdr文件中添加校准信息

    参数:
    hdr_file_path: .hdr文件的路径
    """
    # 要添加的两行信息
    new_lines = [
        "calibration scale factor = 0.10000000149012",
        "data units = W m^-2 sr^-1 um^-1"
    ]

    try:
        # 读取现有的.hdr文件内容
        with open(hdr_file_path, 'r') as f:
            lines = f.readlines()

        # 检查是否已经存在这些行，如果存在则更新，否则添加
        scale_factor_exists = False
        data_units_exists = False

        for i, line in enumerate(lines):
            if line.strip().startswith('calibration scale factor'):
                lines[i] = new_lines[0] + '\n'
                scale_factor_exists = True
            elif line.strip().startswith('data units'):
                lines[i] = new_lines[1] + '\n'
                data_units_exists = True

        # 如果不存在，则在文件末尾添加
        if not scale_factor_exists:
            lines.append(new_lines[0] + '\n')
        if not data_units_exists:
            lines.append(new_lines[1] + '\n')

        # 写回文件
        with open(hdr_file_path, 'w') as f:
            f.writelines(lines)

        print(f"成功更新文件: {hdr_file_path}")

    except Exception as e:
        print(f"处理文件时出错: {e}")


def create_algae_buffer(mask_array, buffer_size=3):
    """
    为藻类掩膜创建缓冲区

    参数:
    mask_array -- 二维numpy数组，藻类区域为1，背景为0
    buffer_size -- 缓冲区大小（像元数），默认为3

    返回:
    扩展后的掩膜数组，藻类核心区+缓冲区都为1
    """
    # 创建圆形结构元素（保证各向同性扩展）
    radius = buffer_size
    size = 2 * radius + 1  # 结构元素尺寸
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    structure = (x ** 2 + y ** 2) <= radius ** 2  # 圆形结构元素

    # 执行形态学膨胀操作
    buffered_mask = ndimage.binary_dilation(
        mask_array.astype(bool),
        structure=structure
    ).astype(np.uint8)

    return buffered_mask



# 构建查找表
#高层云 (162,7,4)
T_altostratus = np.array(extract_MODTRAN(r'Y:\zfr\17-xiaolunwen2\major_revise\submission\data\LUT\高层云计算结果-GF1.txt','T'))
R_altostratus = np.array(extract_MODTRAN(r'Y:\zfr\17-xiaolunwen2\major_revise\submission\data\LUT\高层云计算结果-GF1.txt', 'R'))
T_slope_altostratus = np.array(extract_MODTRAN(r'Y:\zfr\17-xiaolunwen2\major_revise\submission\data\LUT\高层云计算结果-GF1.txt', 'T_slope'))
print(T_slope_altostratus.shape)

# #卷云 (162,7,4)
# T_cirrus = np.array(extract_MODTRAN(r'Y:\zfr\17-xiaolunwen2\major_revise\submission\data\LUT\卷云计算结果-GF1.txt','T'))
# R_cirrus = np.array(extract_MODTRAN(r'Y:\zfr\17-xiaolunwen2\major_revise\submission\data\LUT\卷云计算结果-GF1.txt','R'))
# T_slope_cirrus = np.array(extract_MODTRAN(r'Y:\zfr\17-xiaolunwen2\major_revise\submission\data\LUT\卷云计算结果-GF1.txt','T_slope'))
# print(T_slope_cirrus.shape)

# 定义维度参数
sun_zenith = np.array([10, 20, 30])  # 太阳天顶角
sat_zenith = np.array([0, 10, 20, 30, 40, 50])  # 卫星天顶角
aerosol_od = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])  # 气溶胶光学厚度
cloud_od = np.array([0.06, 0.3, 0.54, 0.78, 1.02, 1.26, 1.5])  # 云光学厚度
wavelengths = np.array([0.485, 0.555, 0.660, 0.830])  # 波长 (nm)


#生成高层云的三个数据的查找表
lookup_table_T_altostratus = create_5d_simulation_data(T_altostratus)
lookup_table_T_altostratus_output_path = r'Y:\zfr\17-xiaolunwen2\major_revise\submission\data\LUT\lookup_table_T_altostratus.nc'
save_lookup_table(lookup_table_T_altostratus, lookup_table_T_altostratus_output_path)

lookup_table_R_altostratus = create_5d_simulation_data(R_altostratus)
lookup_table_R_altostratus_output_path = r'Y:\zfr\17-xiaolunwen2\major_revise\submission\data\LUT\lookup_table_R_altostratus.nc'
save_lookup_table(lookup_table_R_altostratus, lookup_table_R_altostratus_output_path)

lookup_table_T_slope_altostratus = create_5d_simulation_data(T_slope_altostratus)
lookup_table_T_slope_altostratus_output_path = r'Y:\zfr\17-xiaolunwen2\major_revise\submission\data\LUT\lookup_table_T_slope_altostratus.nc'
save_lookup_table(lookup_table_T_slope_altostratus, lookup_table_T_slope_altostratus_output_path)

# #生成卷云的三个数据的查找表
# lookup_table_T_cirrus = create_5d_simulation_data(T_cirrus)
# lookup_table_T_cirrus_output_path = r'Y:\zfr\17-xiaolunwen2\major_revise\submission\data\LUT\lookup_table_T_cirrus.nc'
# save_lookup_table(lookup_table_T_cirrus, lookup_table_T_cirrus_output_path)
#
# lookup_table_R_cirrus = create_5d_simulation_data(R_cirrus)
# lookup_table_R_cirrus_output_path = r'Y:\zfr\17-xiaolunwen2\major_revise\submission\data\LUT\lookup_table_R_cirrus.nc'
# save_lookup_table(lookup_table_R_cirrus, lookup_table_R_cirrus_output_path)
#
# lookup_table_T_slope_cirrus = create_5d_simulation_data(T_slope_cirrus)
# lookup_table_T_slope_cirrus_output_path = r'Y:\zfr\17-xiaolunwen2\major_revise\submission\data\LUT\lookup_table_T_slope_cirrus.nc'
# save_lookup_table(lookup_table_T_slope_cirrus, lookup_table_T_slope_cirrus_output_path)



# 初始化查找表
lookup_T_altostratus = SimpleCloudReflectanceLookup(lookup_table_T_altostratus_output_path)
lookup_R_altostratus = SimpleCloudReflectanceLookup(lookup_table_R_altostratus_output_path)
lookup_T_slope_altostratus = SimpleCloudReflectanceLookup(lookup_table_T_slope_altostratus_output_path)
lookup_altostratus = dict([("T", lookup_T_altostratus), ("R", lookup_R_altostratus), ("T_slope", lookup_T_slope_altostratus)])

# lookup_T_cirrus = SimpleCloudReflectanceLookup(lookup_table_T_cirrus_output_path)
# lookup_R_cirrus = SimpleCloudReflectanceLookup(lookup_table_R_cirrus_output_path)
# lookup_T_slope_cirrus = SimpleCloudReflectanceLookup(lookup_table_T_slope_cirrus_output_path)
# lookup_cirrus = dict([("T", lookup_T_cirrus), ("R", lookup_R_cirrus), ("T_slope", lookup_T_slope_cirrus)])



#输入数据路径
water_class_dir = r'Y:\zfr\17-xiaolunwen2\major_revise\submission\data\water_class image'  #水体分类影像
xml_dir = r'Y:\zfr\17-xiaolunwen2\major_revise\submission\data\Uncorrected image_xml'  #原始影像的元数据.xml
img_dir = r'Y:\zfr\17-xiaolunwen2\major_revise\submission\data\Uncorrected image'  #待校正影像，经过了辐射定标 RPC校正和裁剪
macroalgae_mask_dir = r'Y:\zfr\17-xiaolunwen2\major_revise\submission\data\algae_mask image'  #藻类像元掩膜
HDR_dir = r'Y:\zfr\17-xiaolunwen2\major_revise\submission\data\Uncorrected image_hdr'  #在ENVI中经过辐射定标后的HDR文件
out_toa_dir = r'Y:\zfr\17-xiaolunwen2\major_revise\submission\data\corrected_TOA'  #输出的薄云矫正后TOA反射率影像的文件夹路径
out_rad_dir = r'Y:\zfr\17-xiaolunwen2\major_revise\submission\data\corrected_TOA_to_radiance' #对校正后影像的TOA反射率转换为辐亮度，方便后续大气校正

#AOD年均值
aod = { '2017' :0.465, '2018':0.378, '2019':0.393, '2020':0.437, '2021':0.356, '2022':0.355, '2023':0.396}

#薄云校正
for RGB_img in os.listdir(img_dir):
    if not RGB_img.endswith('.tif'):
        continue
    year = RGB_img.split('_')[4][0:4]
    img_path = os.path.join(img_dir, RGB_img)
    filename = os.path.splitext(RGB_img)[0]
    print(filename,aod[year])

    #搜索xml文件
    for xml in os.listdir(xml_dir):
        if not xml.endswith('.xml'):
            continue
        if (xml.split('_')[4] == filename.split('_')[4]) & (xml.split('_')[5][0:-4] == filename.split('_')[5]):
            xml_path = os.path.join(xml_dir, xml)
            print(xml_path)
            break

    #搜索water_class文件
    for waterclass in os.listdir(water_class_dir):
        if not waterclass.endswith('.tif'):
            continue
        if (waterclass.split('_')[4] == filename.split('_')[4]) & (waterclass.split('_')[5] == filename.split('_')[5]):
            waterclass_path = os.path.join(water_class_dir, waterclass)
            print(waterclass_path)
            break

    #搜索macroalgae_mask文件
    for mask in os.listdir(macroalgae_mask_dir):
        if not mask.endswith('.tif'):
            continue
        if (mask.split('_')[3] == filename.split('_')[4]) & (mask.split('_')[4] == filename.split('_')[5][-4:]):
            mask_path = os.path.join(macroalgae_mask_dir, mask)
            print(mask_path)
            break


    #获取太阳天顶角、卫星观测天顶角、太阳方位角和卫星方位角
    sun_zenith,sate_zenith,sun_azimuth,sate_azimuth = get_obsGeometry(xml_path)
    print(sun_zenith,sate_zenith,sun_azimuth,sate_azimuth)
    #判断是否有耀光,如果存在耀光，则不进行薄云校正
    if abs(sun_azimuth - sate_azimuth)>150:
        print('此文件可能存在耀光')
        #continue

    #利用查找表获取不同云光学厚度下、不同波长的T和R  7*4
    reflectance_matrix_T_altostratus = lookup_altostratus["T"].get_reflectance_matrix(sun_zenith, sate_zenith, aod[year])
    reflectance_matrix_R_altostratus = lookup_altostratus["R"].get_reflectance_matrix(sun_zenith, sate_zenith, aod[year])
    print('altostratus',reflectance_matrix_T_altostratus)
    print('altostratus',reflectance_matrix_R_altostratus)

    #对系数进行拟合
    cloud_extinction = np.array([0.06,0.3,0.54,0.78,1.02,1.26,1.5])

    T1_R1_fit_altostratus = np.polyfit(reflectance_matrix_T_altostratus[:,0], reflectance_matrix_R_altostratus[:,0], 2)
    T1_cloudext_fit_altostratus = np.polyfit(cloud_extinction,reflectance_matrix_T_altostratus[:,0], 2)
    T2_cloudext_fit_altostratus = np.polyfit(cloud_extinction,reflectance_matrix_T_altostratus[:,1], 2)
    T3_cloudext_fit_altostratus = np.polyfit(cloud_extinction,reflectance_matrix_T_altostratus[:,2], 2)
    T4_cloudext_fit_altostratus = np.polyfit(cloud_extinction,reflectance_matrix_T_altostratus[:,3], 2)
    fits_altostratus = np.array([T1_R1_fit_altostratus,T1_cloudext_fit_altostratus,T2_cloudext_fit_altostratus,T3_cloudext_fit_altostratus,T4_cloudext_fit_altostratus])

    fits = dict([("altostratus",fits_altostratus)])  #2个5*3
    print(fits["altostratus"])


    #打开原始影像
    rrs = gdal.Open(img_path)
    band1 = rrs.GetRasterBand(1)
    band2 = rrs.GetRasterBand(2)
    band3 = rrs.GetRasterBand(3)
    band4 = rrs.GetRasterBand(4)
    b1_value = (band1.ReadAsArray()).astype(np.float32)
    b2_value = (band2.ReadAsArray()).astype(np.float32)
    b3_value = (band3.ReadAsArray()).astype(np.float32)
    b4_value = (band4.ReadAsArray()).astype(np.float32)
    print(b1_value)
    #去除背景值以及厚云
    b1_value[(b1_value <= 0.0)|(b2_value <= 0.0)|(b3_value <= 0.0)|(b4_value <= 0.0)|(b3_value >=0.2)] = 0
    b2_value[(b1_value <= 0.0)|(b2_value <= 0.0)|(b3_value <= 0.0)|(b4_value <= 0.0)|(b3_value >=0.2)] = 0
    b3_value[(b1_value <= 0.0)|(b2_value <= 0.0)|(b3_value <= 0.0)|(b4_value <= 0.0)|(b3_value >=0.2)] = 0
    b4_value[(b1_value <= 0.0)|(b2_value <= 0.0)|(b3_value <= 0.0)|(b4_value <= 0.0)|(b3_value >=0.2)] = 0
    #print(b1_value)


    #去除GF-1影像边缘的值，其他卫星数据可忽略该步骤
    pad_width = 1
    b1_value_pad = np.pad(b1_value, ((pad_width, pad_width), (pad_width, pad_width)), 'constant', constant_values=(0, 0))
    row = b1_value_pad.shape[0]
    column = b1_value_pad.shape[1]
    b1_value_right = b1_value_pad[1:row-1,2:column]
    b1_value_left = b1_value_pad[1:row-1,0:column-2]
    b1_value_top = b1_value_pad[0:row-2,1:column-1]
    b1_value_botton = b1_value_pad[2:row,1:column-1]
    judge = (b1_value_right == 0)|(b1_value_left == 0)|(b1_value_top == 0)|(b1_value_botton == 0)
    b1_value[judge] = 0
    b2_value[judge] = 0
    b3_value[judge] = 0
    b4_value[judge] = 0

    #打开藻类掩膜
    rrs_algae_mask = gdal.Open(mask_path)
    band1_algae_mask = rrs_algae_mask.GetRasterBand(1)
    algae_mask_ori = (band1_algae_mask.ReadAsArray())
    algae_mask_ori[algae_mask_ori == 100] = 1
    algae_mask_ori[algae_mask_ori == -100] = 0
    algae_mask_ori[np.isnan(algae_mask_ori)] = 0
    algae_mask_ori[algae_mask_ori < 0] = 0

    #打开水体分类影像
    rrs_class_water = gdal.Open(waterclass_path)
    band1_class_water = rrs_class_water.GetRasterBand(1)
    output_img_water = (band1_class_water.ReadAsArray())
    #获取每类水体的四波段反射率最低值
    labels,labels_mean_b1,labels_mean_b2,labels_mean_b3,labels_mean_b4 = Ca_Cloudfree_Refle(output_img_water,b1_value,b2_value,b3_value,b4_value)
    print(labels,labels_mean_b1,labels_mean_b2,labels_mean_b3,labels_mean_b4)



    #获取原始影像的行和列
    column = band1.XSize
    row = band1.YSize
    print(row,column)


    #计算background_water,也就是藻类像元周围水体像元的反射率
    background_b1 = np.zeros((band1.YSize,band1.XSize),dtype='float32')
    background_b2 = np.zeros((band1.YSize,band1.XSize),dtype='float32')
    background_b3 = np.zeros((band1.YSize,band1.XSize),dtype='float32')
    background_b4 = np.zeros((band1.YSize,band1.XSize),dtype='float32')
    output_T4 = np.zeros((band1.YSize,band1.XSize),dtype='float32')    #计算的云透过率系数CT
    output_T3 = np.zeros((band1.YSize,band1.XSize),dtype='float32')    #计算的云透过率系数CT
    output_T2 = np.zeros((band1.YSize,band1.XSize),dtype='float32')    #计算的云透过率系数CT
    output_T1 = np.zeros((band1.YSize,band1.XSize),dtype='float32')    #计算的云透过率系数CR


    #逐藻类像元校正
    for i in range(row):
        if i% 1000 == 0:
            print(i)
        for j in range(column):
            if (algae_mask_ori[i][j] != 1): #藻类像元为1，其他像元为0
                continue
            else:
                number =0
                wide = 19
                while(number<100):  #确保获取每个藻类像元周边至少100个水体像元的反射率
                    wide = wide+2   #从宽度为19的窗口开始遍历
                    if wide>150:  #如果周围全是厚云，可能水体分类影像全是255，然后wide会取一个很大的值，循环无法终止
                        break
                    center = wide//2
                    top = max(0,i-center)
                    below = min(row,i+center)
                    left = max(0,j-center)
                    right = min(column,j+center)
                    sub_algae = algae_mask_ori[top:below,left:right]
                    sub_label = output_img_water[top:below,left:right]
                    sub_img_b1 = b1_value[top:below,left:right]
                    sub_img_b2 = b2_value[top:below,left:right]
                    sub_img_b3 = b3_value[top:below,left:right]
                    sub_img_b4 = b4_value[top:below,left:right]
                    judge = (sub_algae == 0)&(sub_img_b1>0)
                    number = np.count_nonzero(judge)
                    location = np.where(judge == True)
                if number == 0:
                    continue

                background_b1[i][j] = np.mean(sub_img_b1[location])
                background_b2[i][j] = np.mean(sub_img_b2[location])
                background_b3[i][j] = np.mean(sub_img_b3[location])
                background_b4[i][j] = np.mean(sub_img_b4[location])

                num_class_pixel = len(sub_label[sub_label != 255])  #有效的水体分类像元的数量
                if num_class_pixel < 100:
                    #print(f'{i,j}的有效水体像元个数为0')
                    while (num_class_pixel <100):  # 确保获取每个藻类像元周边至少100个水体像元的反射率
                        wide = wide + 8  #
                        if wide > 200:  # 如果周围全是厚云，可能水体分类影像全是255，然后wide会取一个很大的值，循环无法终止
                            print(f'{i,j}的wide超过了150')
                            break
                        center = wide // 2
                        top = max(0, i - center)
                        below = min(row, i + center)
                        left = max(0, j - center)
                        right = min(column, j + center)
                        sub_label = output_img_water[top:below, left:right]

                        num_class_pixel = len(sub_label[sub_label != 255])



                #获取该藻类像元所在水体的"真值"
                unique_label, counts = np.unique(sub_label, return_counts=True)
                #将unique_label, counts转换为list类型
                unique_label = list(unique_label)
                counts = list(counts)
                if 255 in unique_label:
                    loc_255 = unique_label.index(255)
                    unique_label.remove(255)
                    counts.remove(counts[loc_255])
                if not unique_label:   #如果这个藻类像元周围没有有效的水体像元，则不进行薄云矫正
                    continue

                # ---- 找到出现次数最多的类别 ----
                max_count_idx = counts.index(max(counts))  # 获得最大计数的索引
                best_label = unique_label[max_count_idx]  # 对应的类别标签

                # 找到这个标签在 labels 中的位置
                simulate_indice = labels.index(best_label)

                # 直接取该类别的均值
                simulate_water_b1 = labels_mean_b1[simulate_indice]
                simulate_water_b2 = labels_mean_b2[simulate_indice]
                simulate_water_b3 = labels_mean_b3[simulate_indice]
                simulate_water_b4 = labels_mean_b4[simulate_indice]


                #判断这个藻类像元是否进行薄云矫正
                if (background_b1[i][j] - simulate_water_b1)<0.01:
                    continue


                #进行薄云校正
                T1_roots = np.roots([fits["altostratus"][0][0], simulate_water_b1 +fits["altostratus"][0][1], fits["altostratus"][0][2]-background_b1[i][j]])
                T1 = min(T1_roots)  #T1_roots是个列表, 取最小的那个根
                cloudext_roots = np.roots([fits["altostratus"][1][0],fits["altostratus"][1][1],fits["altostratus"][1][2]-T1])
                cloudext_roots = min(cloudext_roots)  #也是取最小的根
                T4 = fits["altostratus"][4][0]*cloudext_roots*cloudext_roots + fits["altostratus"][4][1]*cloudext_roots +fits["altostratus"][4][2]
                T2 = fits["altostratus"][2][0]*cloudext_roots*cloudext_roots + fits["altostratus"][2][1]*cloudext_roots +fits["altostratus"][2][2]
                T3 = fits["altostratus"][3][0]*cloudext_roots*cloudext_roots + fits["altostratus"][3][1]*cloudext_roots +fits["altostratus"][3][2]

                output_T4[i][j] = T4
                output_T3[i][j] = T3
                output_T2[i][j] = T2
                output_T1[i][j] = T1

                R1 = background_b1[i][j] - simulate_water_b1*T1
                R2 = background_b2[i][j] - simulate_water_b2*T2
                R3 = background_b3[i][j] - simulate_water_b3*T3
                R4 = background_b4[i][j] - simulate_water_b4*T4

                b1_value[i][j] = (b1_value[i][j] - R1)/T1
                b2_value[i][j] = (b2_value[i][j] - R2)/T2
                b3_value[i][j] = (b3_value[i][j] - R3)/T3
                b4_value[i][j] = (b4_value[i][j] - R4)/T4



    #输出薄云校正后的影像
    name = filename + "_thin_clouds_cor.tif"
    output_path = os.path.join(out_toa_dir,name)
    writeTif(output_path,rrs.GetProjection(),rrs.GetGeoTransform(),np.array([b1_value, b2_value, b3_value, b4_value]),6)    #最开始设为空的值还没有恢复

    #输出云透过率系数CT
    # name = filename + "_T.tif"
    # output_path = outputDir + name
    # writeTif(output_path,rrs.GetProjection(),rrs.GetGeoTransform(),np.array([output_T1,output_T2,output_T3,output_T4]),6)    #最开始设为空的值还没有恢复


# 将薄云校正后的TOA反射率转换为辐亮度
for cor_img in os.listdir(out_toa_dir):
    if not cor_img.endswith('.tif'):
        continue
    year = cor_img.split('_')[4][0:4]
    cor_path = os.path.join(out_toa_dir, cor_img)
    filename = os.path.splitext(cor_img)[0]
    print(filename,aod[year])


    # 找到对应的元数据文件.hdr
    for hdr in os.listdir(HDR_dir):
        if not hdr.endswith('.hdr'):
            continue
        if (hdr.split('_')[4] == filename.split('_')[4]) & (hdr.split('_')[5][0:-4] == filename.split('_')[5]):
            hdr_path = os.path.join(HDR_dir, hdr)
            print(hdr_path)
            break

    # 打开原始影像的HDR文件，获取元数据信息
    # 解析头文件
    envi_params = parse_envi_hdr(hdr_path)
    if envi_params:
        # 获取solar irradiance
        if 'solar irradiance' in envi_params:
            Esun = envi_params['solar irradiance']
            # print(gain_values[0])
        else:
            print("在头文件中未找到 'solar irradiance'")

        # 获取Sun_elevation_degree
        if 'sun elevation' in envi_params:
            Sun_elevation_degree = envi_params['sun elevation']
            # print(gain_values[0])
        else:
            print("在头文件中未找到 'Sun_elevation_degree'")

            # 获取影像日期
        if 'acquisition time' in envi_params:
            image_date = envi_params['acquisition time']
            # print(gain_values[0])
        else:
            print("在头文件中未找到 'acquisition time'")

    # 获取日地矫正距离因子
    print(Esun)
    print(Sun_elevation_degree)
    imaging_date = datetime(int(image_date[0:4]), int(image_date[5:7]), int(image_date[8:10]))
    print(imaging_date)
    doy = imaging_date.timetuple().tm_yday  # 获取年积日
    d = 1 + 0.0167 * math.sin(2 * math.pi * (doy - 93.5) / 365)
    print(f"日地距离校正因子 d: {d:.6f}")

    # 打开影像
    rrs = gdal.Open(cor_path)
    band1 = rrs.GetRasterBand(1)
    band2 = rrs.GetRasterBand(2)
    band3 = rrs.GetRasterBand(3)
    band4 = rrs.GetRasterBand(4)

    b1_value = (band1.ReadAsArray())  # 背景值还有厚云在薄云校正时都设为了0
    b2_value = (band2.ReadAsArray())
    b3_value = (band3.ReadAsArray())
    b4_value = (band4.ReadAsArray())
    # FLAASH需要的输入辐亮度单位是： µW/(cm2 * sr * nm)  ，使用公式将表观反射率转换为辐亮度得到的单位是W/(m2 * sr * µm)，因此还需要除以10
    b1_value = b1_value * Esun[0] * math.sin(math.radians(Sun_elevation_degree)) / 3.1415936 / d / d / 10
    b2_value = b2_value * Esun[1] * math.sin(math.radians(Sun_elevation_degree)) / 3.1415936 / d ** 2 / 10
    b3_value = b3_value * Esun[2] * math.sin(math.radians(Sun_elevation_degree)) / 3.1415936 / d ** 2 / 10
    b4_value = b4_value * Esun[3] * math.sin(math.radians(Sun_elevation_degree)) / 3.1415936 / d ** 2 / 10

    output_path = os.path.join(out_rad_dir, filename+ 'to_radiance.dat')   # 替换为你想要的输出路径
    width = rrs.RasterXSize
    height = rrs.RasterYSize
    num_bands = rrs.RasterCount
    data_type = rrs.GetRasterBand(1).DataType
    driver = gdal.GetDriverByName('ENVI')
    out_ds = driver.Create(output_path, width, height, num_bands, data_type, ['INTERLEAVE=BIL'])
    out_ds.SetProjection(rrs.GetProjection())
    out_ds.SetGeoTransform(rrs.GetGeoTransform())

    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(b1_value)
    out_band = out_ds.GetRasterBand(2)
    out_band.WriteArray(b2_value)
    out_band = out_ds.GetRasterBand(3)
    out_band.WriteArray(b3_value)
    out_band = out_ds.GetRasterBand(4)
    out_band.WriteArray(b4_value)
    del out_ds

    copy_hdr_sections(
        hdr_path,
        os.path.join(out_rad_dir, filename+ 'to_radiance.hdr'),
        ['wavelength', 'fwhm', 'wavelength units', 'data ignore value',
         'data gain values', 'data offset values', 'sun elevation',
         'solar irradiance', 'sensor type', 'acquisition time']
    )
    add_hdr_metadata(os.path.join(out_rad_dir, filename+ 'to_radiance.hdr'))
