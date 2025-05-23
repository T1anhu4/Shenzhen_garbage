from ultralytics import YOLO

model = YOLO('yolov8l.pt')

model.train(
    data = '/home/zhy/wth/wth3/YOLODataset/dataset.yaml',
    epochs = 400,
    imgsz = (1536, 864),  # 增大分辨率
    batch = 6,  # 因分辨率增大适当降低batch
    workers = 6,
    device = 0,
    optimizer = 'SGD',
    lr0 = 0.01,  # 提高初始学习率
    lrf = 0.2,  # 减缓衰减速度
    momentum = 0.9,
    weight_decay = 0.0003,  # 降低正则化强度
    warmup_epochs = 20.0,
    patience = 100,  # 延长早停等待
    warmup_momentum = 0.7,
    warmup_bias_lr = 0.2,
    box = 5.0,  # 调整损失权重平衡
    cls = 3.0,  # 强化分类损失
    dfl = 2.0,
    kobj = 2.0,  # 加强目标存在性学习
    hsv_h = 0.5,  # 增强色调扰动
    hsv_s = 0.9,  # 提高饱和度扰动
    hsv_v = 0.9,
    degrees = 15.0,  # 增加旋转角度
    shear = 12.0,  # 加强剪切形变
    perspective = 0.002,
    mixup = 0.3,  # 增强混合增强
    copy_paste = 0.4,  # 提高复制粘贴概率
    close_mosaic = 50,  # 延后关闭马赛克
    label_smoothing = 0.1,  # 引入标签平滑
    amp = True
)
