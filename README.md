# Yolov8-cls-Inference
Yolov8分类模型的推理


结论：
    实验发现在相同的条件下这个opencv的Resize 和 PIL的Resize 得到的最终结果图有所不同，这回导致在后续推理的结果置信度有所偏差。
    这种影响在目标检测中并不明显，但是在分类任务中差异较大。