import numpy as np

class SimpleNearestNeighbor:
    """
    简单最近邻分类器
    使用绝对值距离（曼哈顿距离）
    """
    
    def __init__(self):
        # 存储训练数据
        self.X_train = None  # 训练特征
        self.y_train = None  # 训练标签
    
    def train(self, X_train, y_train):
        """
        训练阶段：只是存储数据，不做复杂计算
        时间复杂度：O(1) - 只是存储引用，不随数据量增加而变化
        
        参数:
        X_train: 训练特征数据
        y_train: 训练标签数据
        """
        print("开始训练...")
        print(f"训练数据形状: {X_train.shape}")
        
        # 只是简单存储数据，没有复杂的模型构建过程
        self.X_train = X_train
        self.y_train = y_train
        
        print("训练完成！只是存储了数据，没有进行计算\n")
    
    def predict(self, X_test):
        """
        测试阶段：对每个测试点，找到最近的训练点
        时间复杂度：O(N) - 需要遍历所有训练数据
        
        参数:
        X_test: 测试特征数据
        
        返回:
        predictions: 预测结果
        """
        print("开始预测...")
        print(f"测试数据形状: {X_test.shape}")
        print(f"需要比较的训练数据数量: {len(self.X_train)}")
        
        predictions = []  # 存储预测结果
        
        # 对每个测试点进行预测
        for i, test_point in enumerate(X_test):
            print(f"\n--- 预测第 {i+1} 个测试点 ---")
            print(f"测试点: {test_point}")
            
            min_distance = float('inf')  # 初始化最小距离为无穷大
            best_index = -1  # 最近邻居的索引
            
            # 遍历所有训练点，找到最近的那个
            for j, train_point in enumerate(self.X_train):
                # 计算绝对值距离（曼哈顿距离）
                distance = self._absolute_distance(test_point, train_point)
                
                print(f"  与训练点 {j} ({train_point}) 的距离: {distance}")
                
                # 如果找到更近的点，更新最小距离和索引
                if distance < min_distance:
                    min_distance = distance
                    best_index = j
            
            # 使用最近邻居的标签作为预测结果
            predicted_label = self.y_train[best_index]
            predictions.append(predicted_label)
            
            print(f"✓ 最近邻居是训练点 {best_index}, 标签: {predicted_label}")
            print(f"✓ 最小距离: {min_distance}")
            print(f"✓ 预测结果: {predicted_label}")
        
        print(f"\n预测完成！总共处理了 {len(X_test)} 个测试点")
        return np.array(predictions)
    
    def _absolute_distance(self, point1, point2):
        """
        计算两个点之间的绝对值距离
        对于每个特征，计算绝对差值，然后求和
        
        参数:
        point1, point2: 两个数据点
        
        返回:
        distance: 绝对值距离
        """
        # 使用numpy的绝对值函数，然后求和
        return np.sum(np.abs(point1 - point2))

# ==================== 示例使用 ====================

# 创建简单的训练数据
# 假设我们有4个训练样本，每个样本有2个特征
X_train = np.array([
    [1, 2],    # 样本1
    [2, 3],    # 样本2  
    [3, 1],    # 样本3
    [4, 4]     # 样本4
])

# 对应的标签（0代表类别A，1代表类别B）
y_train = np.array([0, 0, 1, 1])

print("训练数据:")
for i, (features, label) in enumerate(zip(X_train, y_train)):
    print(f"样本{i}: 特征={features}, 标签={'A' if label == 0 else 'B'}")

print("\n" + "="*50 + "\n")

# 创建分类器实例
nn_classifier = SimpleNearestNeighbor()

# 训练阶段 - O(1) 时间复杂度
nn_classifier.train(X_train, y_train)

print("="*50 + "\n")

# 创建测试数据
X_test = np.array([
    [1.5, 2.5],  # 测试点1 - 应该接近类别A
    [3.5, 3.5]   # 测试点2 - 应该接近类别B
])

print("测试数据:")
for i, test_point in enumerate(X_test):
    print(f"测试点{i+1}: {test_point}")

print("\n" + "="*50 + "\n")

# 测试阶段 - O(N) 时间复杂度  
predictions = nn_classifier.predict(X_test)

print("\n" + "="*50)
print("最终预测结果:")
for i, (test_point, pred) in enumerate(zip(X_test, predictions)):
    print(f"测试点{i+1} {test_point} -> 预测类别: {'A' if pred == 0 else 'B'}")