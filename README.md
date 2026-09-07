# Oxford‑IIIT‑Pet 宠物分类项目
基于ResNet18实现37类宠物图像分类任务

## 实验指标
- 数据集：Oxford‑IIIT‑Pet，共37类宠物
- 验证集最佳准确率：**93.30%**
- 测试集准确率：**85.47%**
> 注：验证集与测试集准确率存在明显差距，模型存在一定过拟合现象。
- 训练权重输出目录：`checkpoint/best.pth`

## 数据集准备
使用 Oxford‑IIIT‑Pet 数据集，共37类宠物。
将数据集解压放到如下路径：`./data/petdata/oxford‑iiit‑pet/`

> ⚠重要：解压完成后，`./data/petdata/oxford‑iiit‑pet/` 目录下直接看到 `images`、`annotations` 两个文件夹，不要多层嵌套。
> 错误示例：`data/petdata/oxford‑iiit‑pet/oxford‑iiit‑pet/images`（多嵌套一层会报找不到数据集）

💡提示：Windows本地调试若报数据集找不到，可临时改为本机绝对路径；提交代码必须使用相对路径，保证Linux/Colab环境可复现。

```plaintext
./data/petdata/oxford‑iiit‑pet/
├─ annotations
└─ images
```

## 项目文件结构

```
├── models/model.py        # ResNet18模型定义
├── utils/metrics.py      # 评价指标、混淆矩阵
├── utils/gradcam.py      # Grad‑CAM热力图可视化
├── train.py              # 训练入口脚本
├── evaluate.py           # 独立评估脚本
├── requirements.txt      # 依赖包列表
└── README.md             # 项目说明文档
```

## 运行完整流程

```
# 安装依赖
pip install -r requirements.txt

# 模型训练
python train.py

# 模型评估
python evaluate.py
```

### 实验结果
混淆矩阵：
![混淆矩阵](test_cm.png)

Grad‑CAM热力图可视化：

**样本1：预测正确 (Correct | GT=0)**
![预测正确样本热力图](gradcam_correct.png)

**样本2：预测错误 (Wrong | GT=0, Pred=32)**
![预测错误样本热力图](gradcam_wrong.png)

#### 结果分析
Grad‑CAM热力图可以直观观察模型关注的图像区域。
预测正确样本中，热力高亮区域覆盖猫咪头部与身体主体，模型能够捕捉目标的关键特征，完成正确分类。
预测错误样本中，热力仅集中在耳朵、口鼻等局部区域，模型没有充分提取完整的目标特征，特征提取出现偏差，最终导致分类错误。






