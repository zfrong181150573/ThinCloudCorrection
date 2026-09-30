import numpy as np
from osgeo import gdal
import os
from scipy import ndimage
from sklearn.cluster import KMeans, MiniBatchKMeans
import time



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


def indices_of_smallest_10_percent(arr, per):
    """
    返回数组中前10%最小值的索引

    参数:
    arr: 输入数组（NumPy数组或类似数组的结构）

    返回:
    包含前10%最小值索引的数组
    """
    arr = np.asarray(arr)
    n = len(arr)
    if n == 0:
        return np.array([], dtype=int)

    # 计算需要选取的元素数量（向上取整）
    k = max(1, int(np.ceil(n * per)))

    # 使用argpartition高效获取前k个最小值的索引
    # 注意：返回的索引顺序不一定与值的大小顺序一致
    indices = np.argpartition(arr, k)[:k]

    # 如果需要按值的大小排序索引
    # 获取这些索引对应的值
    values = arr[indices]
    # 对值进行排序并获取排序后的索引顺序
    sorted_order = np.argsort(values)
    # 按值的大小顺序返回索引
    return indices[sorted_order]



# 利用kmeans直接将影像分类，迭代k值并且使用minibatch，提高速度，查看sse,正式最终版

thresholds_sum = []
k_best = []
sse_end = []
sse_k = []
p_value = [0]


img_dir = r'Y:\zfr\17-xiaolunwen2\major_revise\submission\data\Uncorrected image'  #输入影像文件夹路径
outputDir = r'Y:\zfr\17-xiaolunwen2\major_revise\submission\data\water_class image'  #输出文件夹路径
wavelength = {'B1': 0.485,'B2': 0.555, 'B3': 0.675, 'B4': 0.789}  #GF-1影像各波段波长
for img in os.listdir(img_dir):
    if not(img.endswith('.tif')):
        continue
    img_path = os.path.join(img_dir, img)
    filename = os.path.splitext(img)[0]
    print(filename)

    rrs = gdal.Open(img_path)

    band1 = rrs.GetRasterBand(1)
    band2 = rrs.GetRasterBand(2)
    band3 = rrs.GetRasterBand(3)
    band4 = rrs.GetRasterBand(4)

    b1_value = (band1.ReadAsArray()).astype(np.float32)
    b2_value = (band2.ReadAsArray()).astype(np.float32)
    b3_value = (band3.ReadAsArray()).astype(np.float32)
    b4_value = (band4.ReadAsArray()).astype(np.float32)
    print(b1_value.shape)

    # 去除背景值以及厚云
    b1_value[(b1_value <= 0.0) | (b2_value <= 0.0) | (b3_value <= 0.0) | (b4_value <= 0.0) | (b3_value >= 0.2)] = np.nan
    b2_value[(b1_value <= 0.0) | (b2_value <= 0.0) | (b3_value <= 0.0) | (b4_value <= 0.0) | (b3_value >= 0.2)] = np.nan
    b3_value[(b1_value <= 0.0) | (b2_value <= 0.0) | (b3_value <= 0.0) | (b4_value <= 0.0) | (b3_value >= 0.2)] = np.nan
    b4_value[(b1_value <= 0.0) | (b2_value <= 0.0) | (b3_value <= 0.0) | (b4_value <= 0.0) | (b3_value >= 0.2)] = np.nan


    #去除GF-1影像边缘的值，其他影像可省略这一步骤
    pad_width = 1
    b1_value_pad = np.pad(b1_value, ((pad_width, pad_width), (pad_width, pad_width)), 'constant',
                          constant_values=(np.nan, np.nan))
    row = b1_value_pad.shape[0]
    column = b1_value_pad.shape[1]
    b1_value_right = b1_value_pad[1:row - 1, 2:column]
    b1_value_left = b1_value_pad[1:row - 1, 0:column - 2]
    b1_value_top = b1_value_pad[0:row - 2, 1:column - 1]
    b1_value_botton = b1_value_pad[2:row, 1:column - 1]
    judge = (np.isnan(b1_value_right) | np.isnan(b1_value_left) | np.isnan(b1_value_top) | np.isnan(
        b1_value_botton))
    b1_value[judge] = np.nan
    b2_value[judge] = np.nan
    b3_value[judge] = np.nan
    b4_value[judge] = np.nan

    # 设置藻类区域，并建立3个像元的缓冲区
    algae_mask_ori = np.zeros(b1_value.shape)
    algae_mask_ori[b4_value > b3_value - 0.005] = 1
    algae_mask_exp = create_algae_buffer(algae_mask_ori, buffer_size=3)
    b1_value[algae_mask_exp == 1] = np.nan
    b2_value[algae_mask_exp == 1] = np.nan
    b3_value[algae_mask_exp == 1] = np.nan
    b4_value[algae_mask_exp == 1] = np.nan

    # 计算两两波段差值
    slope_b43 = b4_value - b3_value
    slope_b32 = b3_value - b2_value
    slope_b21 = b2_value - b1_value

    # 输出影像
    output_img = np.zeros(b1_value.shape)

    nan_loca = np.where((np.isnan(slope_b43)) | (np.isnan(slope_b32)) | (np.isnan(slope_b21)))
    nonan_loca = np.where((~np.isnan(slope_b43)) & (~np.isnan(slope_b32)) & (~np.isnan(slope_b21)))

    output_img[nan_loca] = 255  # 空值设为255
    slopeb43_p = slope_b43[(~np.isnan(slope_b43)) & (~np.isnan(slope_b32)) & (~np.isnan(slope_b21))]
    slopeb32_p = slope_b32[(~np.isnan(slope_b43)) & (~np.isnan(slope_b32)) & (~np.isnan(slope_b21))]
    slopeb21_p = slope_b21[(~np.isnan(slope_b43)) & (~np.isnan(slope_b32)) & (~np.isnan(slope_b21))]

    X = np.concatenate((slopeb43_p.reshape((-1, 1)), slopeb32_p.reshape((-1, 1)), slopeb21_p.reshape((-1, 1))),axis=1)
    print(X.shape)

    start_time = time.time()

    # 2. 分层抽样 - 使用MiniBatchKMeans创建分层
    print("\n--- 开始分层抽样 ---")
    n_strata = 20  # 层数，最多50层
    print(f"使用 {n_strata} 个分层进行抽样")

    # 使用MiniBatchKMeans进行快速预聚类
    pre_cluster = MiniBatchKMeans(n_clusters=n_strata, random_state=42,
                                  n_init=5, batch_size=100000)
    strata_labels = pre_cluster.fit_predict(X)
    sample_ratio = 0.2

    # 分层抽样
    unique_strata, counts = np.unique(strata_labels, return_counts=True)
    sample_per_stratum = np.round(counts * sample_ratio).astype(int)
    sample_per_stratum = np.maximum(sample_per_stratum, 1)  # 确保每层至少一个样本

    sampled_indices = []
    for stratum, n_samples in zip(unique_strata, sample_per_stratum):
        stratum_indices = np.where(strata_labels == stratum)[0]
        if len(stratum_indices) > n_samples:
            selected = np.random.choice(stratum_indices, n_samples, replace=False)
        else:
            selected = stratum_indices
        sampled_indices.extend(selected)

    sampled_data = X[sampled_indices]
    sampling_time = time.time()
    print(f"分层抽样完成，抽样数据形状: {sampled_data.shape}")
    print(f"抽样耗时: {sampling_time - start_time:.2f}秒")

    # 3. 在抽样数据上确定最佳K值
    print("\n--- 在抽样数据上确定最佳K值 ---")
    sse = []
    k_range = range(1, 12)  #最多分10类
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
        kmeans.fit(sampled_data)

        sse.append(kmeans.inertia_)
        print(sse)
        # 判断是否停止分类  #到时候恢复
        if (k > 1) & ((sse[k - 2] - sse[k - 1]) < 50):
            break

    print(f'使用样本进行分类的最佳K值为：{k-1}')

    sample_best_k = k - 1
    kmeans = KMeans(n_clusters=sample_best_k, random_state=42, n_init=3, verbose=0, max_iter=100)   #verbose控制程序在运行过程中是否输出详细的进度日志信息。0是不显示
    kmeans.fit(X)
    print('分类完成')

    X_labels = np.array(kmeans.labels_)
    print(X_labels.shape)

    output_img[nonan_loca] = X_labels

    # 计算每个标签的平均反射率
    labels_unique = list(np.unique(output_img))
    print(labels_unique)
    labels_unique.remove(255)

    #修正清洁水体的分类
    for la in labels_unique:
        slope_b43_class = slope_b43[output_img == la]
        slope_b32_class = slope_b32[output_img == la]
        slope_b21_class = slope_b21[output_img == la]

        MDRGB = np.median((slope_b32_class / (wavelength['B3'] - wavelength['B2'])) - (slope_b21_class / (wavelength['B2'] - wavelength['B1'])))
        MDIGR = np.median((slope_b43_class / (wavelength['B4'] - wavelength['B3'])) - (slope_b32_class / (wavelength['B3'] - wavelength['B2'])))
        SIR= np.median(slope_b43_class)
        SRG = np.median(slope_b32_class)
        SGB = np.median(slope_b21_class)
        # X_means.append(X_mean)
        print(la, SGB, SRG,SIR, MDRGB,MDIGR)
        if ((SGB< 0) & (SRG < 0) & (SIR < 0) & (MDRGB > 0.1) & (MDIGR > 0)):  # 说明是清水
            print(la)
            output_img[output_img == la] = 10

    #再次显示分类类别
    labels_unique = list(np.unique(output_img))
    labels_unique.remove(255)
    print(labels_unique)
    outname = filename + "_class_kmeans_" + str(len(labels_unique)) + ".tif"
    output_path = os.path.join(outputDir,outname)
    writeTif(output_path, rrs.GetProjection(), rrs.GetGeoTransform(), output_img, 1)